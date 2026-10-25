
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from CMC_utils.models.tabular.naim.naim import NAIM

from config import (
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    DEVICE,
    NAIM_MAX_EPOCHS,
    NAIM_PATIENCE,
    NAIM_BATCH_SIZE,
    NAIM_LEARNING_RATE,
    NAIM_D_TOKEN,
    NAIM_NUM_HEADS,
    NAIM_FEEDFORWARD_DIM,
    NAIM_DROPOUT_RATE,
    NAIM_NUM_LAYERS,
)


def augment_training_batch(batch_X):
    """
    Apply training-time random masking regularization.

    Each sample has a 50% probability of being selected.
    If selected, randomly mask between 1 and v-1 observed
    features, where v is the number of observed features.

    At least one observed feature is preserved.
    Original missing values remain missing.
    """
    augmented_X = batch_X.clone()

    for i in range(augmented_X.shape[0]):

        # Select this sample with probability 0.5.
        if torch.rand(1).item() >= 0.5:
            continue

        # Find features that are currently observed.
        observed_indices = torch.where(
            ~torch.isnan(augmented_X[i])
        )[0]

        v = len(observed_indices)

        # Preserve at least one observed feature.
        if v <= 1:
            continue

        # Randomly choose c from {1, 2, ..., v-1}.
        c = torch.randint(
            low=1,
            high=v,
            size=(1,),
        ).item()

        # Choose c observed features without replacement.
        permutation = torch.randperm(v)
        selected_indices = observed_indices[
            permutation[:c]
        ]

        # Hide selected features for this training step.
        augmented_X[i, selected_indices] = float("nan")

    return augmented_X


def prepare_naim_data(X_train, X_val, X_test):
    """
    Prepare data for NAIM.

    Numeric columns are min-max scaled using training data only.
    Categorical values are encoded using categories observed in
    training data only. Missing and unseen values remain NaN.
    """

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # Scale numeric features using training-set statistics only.
    for column in NUMERIC_COLUMNS:
        train_values = X_train[column].to_numpy(dtype=float)
        val_values = X_val[column].to_numpy(dtype=float)
        test_values = X_test[column].to_numpy(dtype=float)

        # If the entire training column is missing, leave it missing.
        if np.all(np.isnan(train_values)):
            continue

        min_value = np.nanmin(train_values)
        max_value = np.nanmax(train_values)
        scale = max_value - min_value

        if scale > 0:
            X_train[column] = (train_values - min_value) / scale
            X_val[column] = (val_values - min_value) / scale
            X_test[column] = (test_values - min_value) / scale
        else:
            # Constant observed training column.
            X_train[column] = np.where(
                np.isnan(train_values), np.nan, 0.0
            )
            X_val[column] = np.where(
                np.isnan(val_values), np.nan, 0.0
            )
            X_test[column] = np.where(
                np.isnan(test_values), np.nan, 0.0
            )

    # Encode categorical columns.
    # Categories are learned from training data only.
    cat_idxs = []
    cat_dims = []

    for column in CATEGORICAL_COLUMNS:
        train_values = X_train[column].astype(object)
        val_values = X_val[column].astype(object)
        test_values = X_test[column].astype(object)

        categories = sorted(
            train_values.dropna().astype(str).unique().tolist()
        )

        category_to_id = {
            category: index
            for index, category in enumerate(categories)
        }

        def encode(values):
            encoded = values.map(
                lambda value: (
                    category_to_id.get(str(value), np.nan)
                    if pd.notna(value)
                    else np.nan
                )
            )
            return encoded.astype(float)

        X_train[column] = encode(train_values)
        X_val[column] = encode(val_values)
        X_test[column] = encode(test_values)

        cat_idxs.append(X_train.columns.get_loc(column))

        # NAIM requires a valid category dimension.
        cat_dims.append(max(len(categories), 1))

    # Put columns in the same order for all splits.
    X_train_array = X_train.to_numpy(dtype=float)
    X_val_array = X_val.to_numpy(dtype=float)
    X_test_array = X_test.to_numpy(dtype=float)

    return (
        X_train_array,
        X_val_array,
        X_test_array,
        cat_idxs,
        cat_dims,
    )


def train_and_predict_naim(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    seed,
):
    """
    Train NAIM with validation Macro-F1 early stopping.
    Return test predictions and training diagnostics.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    (
        X_train_array,
        X_val_array,
        X_test_array,
        cat_idxs,
        cat_dims,
    ) = prepare_naim_data(X_train, X_val, X_test)

    X_train_tensor = torch.tensor(
        X_train_array,
        dtype=torch.float32,
        device=DEVICE,
    )

    X_val_tensor = torch.tensor(
        X_val_array,
        dtype=torch.float32,
        device=DEVICE,
    )

    y_train_tensor = torch.tensor(
        np.asarray(y_train),
        dtype=torch.long,
        device=DEVICE,
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=NAIM_BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = NAIM(
        input_size=X_train_array.shape[1],
        output_size=2,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        d_token=NAIM_D_TOKEN,
        embedder_initialization="normal",
        bias=False,
        mask_type=2,
        missing_value="~inf",
        num_heads=NAIM_NUM_HEADS,
        feedforward_dim=NAIM_FEEDFORWARD_DIM,
        dropout_rate=NAIM_DROPOUT_RATE,
        activation="relu",
        num_layers=NAIM_NUM_LAYERS,
        extractor=False,
    ).to(DEVICE)

    # Calculate class weights from training labels only.
    class_counts = np.bincount(
        np.asarray(y_train, dtype=int),
        minlength=2,
    )

    class_weights = np.zeros(2, dtype=np.float32)
    present_classes = class_counts > 0

    class_weights[present_classes] = (
        len(y_train)
        / (
            int(present_classes.sum())
            * class_counts[present_classes]
        )
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=NAIM_LEARNING_RATE,
    )

    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    best_train_f1 = 0.0
    best_val_f1 = -1.0
    best_epoch = 0

    for epoch in range(NAIM_MAX_EPOCHS):
        model.train()

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            # Apply regularization only to training inputs.
            # The original batch remains unchanged.
            batch_X_augmented = augment_training_batch(batch_X)

            logits = model(batch_X_augmented)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

        # Evaluate on the original, unaugmented training data
        # and validation data.
        model.eval()

        with torch.no_grad():
            train_logits = model(X_train_tensor)
            val_logits = model(X_val_tensor)

            train_predictions = (
                train_logits.argmax(dim=1).cpu().numpy()
            )

            val_predictions = (
                val_logits.argmax(dim=1).cpu().numpy()
            )

        train_f1 = f1_score(
            y_train,
            train_predictions,
            average="macro",
            zero_division=0,
        )

        val_f1 = f1_score(
            y_val,
            val_predictions,
            average="macro",
            zero_division=0,
        )

        print(
            f"Epoch {epoch + 1:02d}/{NAIM_MAX_EPOCHS} | "
            f"Train Macro-F1: {train_f1:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_train_f1 = train_f1
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= NAIM_PATIENCE:
            print("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError("NAIM training did not produce a best model.")

    # Restore the checkpoint with the best validation Macro-F1.
    model.load_state_dict(best_state)
    model.eval()

    X_test_tensor = torch.tensor(
        X_test_array,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        test_logits = model(X_test_tensor)

        predictions = (
            test_logits.argmax(dim=1).cpu().numpy()
        )

    return (
        predictions,
        best_train_f1,
        best_val_f1,
        best_epoch,
    )
