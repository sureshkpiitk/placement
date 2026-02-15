import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Optional



def prepare_features(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """Cleans dataframe, does simple encoding and returns train/test splits.

    Returns: X_train, X_test, y_train, y_test, preprocessor (Pipeline)
    """
    df = df.drop(labels=["Student_ID", "salary_lpa"], axis=1)  # drop completely empty columns
    df = df.copy()
    print("Initial dataframe shape:", df.shape)

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    # Basic cleaning: strip strings
    for c in df.select_dtypes(include=[object]).columns:
        df[c] = df[c].astype(str).str.strip()

    # Convert target to binary: 'placed' -> 1, everything else -> 0
    y = df[target_col].astype(str).str.strip().str.lower().eq('placed').astype(int)
    X = df.drop(columns=[target_col])

    # Identify categorical and numeric
    cat_cols = X.select_dtypes(include=[object, "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Column transformer
    preprocessor = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ], remainder="passthrough")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique())>1 else None)

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    import sys
    from .data_loader import load_data
    p = sys.argv[1] if len(sys.argv) > 1 else "data/Placement_Data_Full_Class.csv"
    df = load_data(p)
    tc = infer_target_column(df)
    print("Inferred target:", tc)
