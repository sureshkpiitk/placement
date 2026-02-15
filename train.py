import argparse
from data_loader import load_data
from preprocess import prepare_features
from models import train_models, pick_best, save_model, save_metrics
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_col", default="placement_status", help="Target column name. If omitted, inferred")
    parser.add_argument("--model_out", default="best_model.joblib")
    parser.add_argument("--eval_out", default="evaluation.json")
    args = parser.parse_args()

    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = prepare_features(df, target_col=args.target_col)

    results = train_models(preprocessor, X_train, X_test, y_train, y_test)
    print("colums", X_test.columns)
    # print("colums", y_test.columns)
    test_dataset = pd.concat([X_test, y_test], axis=1)
    test_dataset.to_csv("data/test_dataset.csv", index=False)

    best_name, best_model, best_metrics = pick_best(results)

    # Save
    save_model(best_model, args.model_out)
    out = {
        'best_model': best_name,
        'metrics': best_metrics,
        'all_models': {k: v['metrics'] for k, v in results.items()}
    }
    save_metrics(out, args.eval_out)
    print("Training complete. Best model:", best_name)
    print("Model saved to:", args.model_out)
    print("Evaluation saved to:", args.eval_out)


if __name__ == "__main__":
    main()
