import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TRAIN_PATH = "student_train_full.csv"
TEST_PATH = "public_test_questions.csv"
OUTPUT_PATH = "cys_day2_focus_preds.csv"
RANDOM_STATE = 42


def clean_price_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df.loc[df["price"].isin(["", "nan", "None"]), "price"] = pd.NA
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    return df


def add_host_since_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    host_since_dt = pd.to_datetime(df["host_since"], errors="coerce")
    ref_date = pd.Timestamp("2025-09-14")
    df["host_tenure_days"] = (ref_date - host_since_dt).dt.days
    return df


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = clean_price_column(train_df)
    train_df = add_host_since_features(train_df)
    test_df = add_host_since_features(test_df)

    target = "price"
    focused_features = [
        "property_type",
        "bathrooms",
        "latitude",
        "longitude",
        "host_tenure_days",
        "number_of_reviews",
        "review_scores_value",
    ]

    X = train_df[focused_features].copy()
    y = train_df[target].copy()
    X_test = test_df[focused_features].copy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    numeric_features = [
        "bathrooms",
        "latitude",
        "longitude",
        "host_tenure_days",
        "number_of_reviews",
        "review_scores_value",
    ]
    categorical_features = ["property_type"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=16,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    # 价格长尾明显，log 训练更稳
    pipe.fit(X_train, np.log1p(y_train))
    valid_pred = np.expm1(pipe.predict(X_valid))

    mae = mean_absolute_error(y_valid, valid_pred)
    r2 = r2_score(y_valid, valid_pred)

    print("===== Focused Feature Model =====")
    print(f"Features: {focused_features}")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")

    test_pred = np.expm1(pipe.predict(X_test))
    out_df = pd.DataFrame({"id": test_df["id"], "predicted_price": test_pred})
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
