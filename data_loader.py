import io
import zipfile
import requests
import pandas as pd
from pathlib import Path


KAGGLE_ZIP_URL = "https://www.kaggle.com/api/v1/datasets/download/vishardmehta/indian-engineering-college-placement-dataset"


def _download_zip_to_bytes(url: str) -> bytes:
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    return resp.content


def load_data(
    path: str = "data/Placement_Data_Full_Class.csv",
    download_url: str = KAGGLE_ZIP_URL,
    prefer_local: bool = True,
    merge_targets: bool = False,
) -> pd.DataFrame:
    """Load dataset: prefer local CSV; otherwise download ZIP via `requests`, unzip and read CSV.

    Args:
        path: Local desired CSV path.
        merge_targets: If True and `placement_targets.csv` is inside the ZIP or data/, merge targets.

    Returns:
        pandas.DataFrame
    """
    data_dir = Path("data")
    csv_path = Path(path)

    # Try local file first
    if prefer_local and csv_path.exists():
        return pd.read_csv(csv_path)

    # Download zip bytes
    try:
        b = _download_zip_to_bytes(download_url)
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset from {download_url}: {e}") from e

    # Read Zip from bytes and find CSVs
    z = zipfile.ZipFile(io.BytesIO(b))
    csv_candidates = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if not csv_candidates:
        raise RuntimeError("No CSV files found in downloaded ZIP")

    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Extract and save all CSVs to data/, read into dataframes
    dfs = {}
    for name in csv_candidates:
        out_path = data_dir / Path(name).name
        with z.open(name) as fh:
            content = fh.read()
            out_path.write_bytes(content)
        try:
            dfs[name] = pd.read_csv(out_path)
        except Exception:
            # skip unreadable
            continue

    # If no readable CSVs
    if not dfs:
        raise RuntimeError("No readable CSV files found in downloaded ZIP")

    # Identify main (features) and target files
    main_name = None
    target_name = None
    # prefer known names
    for name in dfs:
        lname = name.lower()
        if "indian_engineering_student_placement" in lname or "placement_data" in lname:
            main_name = name
        if "placement_targets" in lname or "placement_target" in lname or "target" in lname:
            target_name = name

    # heuristics if not found
    if main_name is None:
        # choose the CSV with more columns as main
        main_name = max(dfs.keys(), key=lambda n: dfs[n].shape[1])
    if target_name is None and len(dfs) >= 2:
        # choose the CSV with fewer columns as target
        target_name = min(dfs.keys(), key=lambda n: dfs[n].shape[1])

    df_main = dfs[main_name]

    if target_name and target_name in dfs:
        df_t = dfs[target_name]
        # attempt join on common keys
        join_key = None
        for key in ("sl_no", "Sl_no", "student_id", "id"):
            if key in df_main.columns and key in df_t.columns:
                join_key = key
                break
        if join_key:
            df = df_main.merge(df_t, on=join_key, how="left")
        else:
            # if lengths align, concat side-by-side
            if len(df_main) == len(df_t):
                df = pd.concat([df_main.reset_index(drop=True), df_t.reset_index(drop=True)], axis=1)
            else:
                df = df_main
    else:
        df = df_main

    return df


if __name__ == "__main__":
    df = load_data()
    print("Loaded dataframe shape:", df.shape)

