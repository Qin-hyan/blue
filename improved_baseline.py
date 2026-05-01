import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


TRAIN_PATH = "student_train_full.csv"
TEST_PATH = "public_test_questions.csv"
OUTPUT_PRED_PATH = "public_test_predictions_improved.csv"
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


def drop_unhelpful_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    full_null_cols = df.columns[df.isna().all()].tolist()
    clearly_irrelevant = [
        "id",
        "listing_url",
        "scrape_id",
        "last_scraped",
        "source",
        "name",
        "description",
        "neighborhood_overview",
        "picture_url",
        "host_id",
        "host_url",
        "host_name",
        "host_since",
        "host_location",
        "host_about",
        "host_thumbnail_url",
        "host_picture_url",
        "host_verifications",
        "bathrooms_text",
        "amenities",
        "calendar_updated",
        "calendar_last_scraped",
        "first_review",
        "last_review",
        "license",
    ]
    to_drop = sorted(set(full_null_cols + clearly_irrelevant))
    return df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = clean_price_column(train_df)
    train_df = drop_unhelpful_columns(train_df)
    test_df = drop_unhelpful_columns(test_df)

    # 保证训练/测试特征列对齐，并仅使用数值特征
    target = "price"
    feature_cols = [c for c in train_df.columns if c != target and c in test_df.columns]
    X = train_df[feature_cols].select_dtypes(include=["number"]).copy()
    y = train_df[target].copy()
    X_test = test_df[X.columns].copy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # 删除训练集中全空的特征列，避免中位数仍为空
    all_null_cols = [c for c in X_train.columns if X_train[c].isna().all()]
    if all_null_cols:
        X_train = X_train.drop(columns=all_null_cols)
        X_valid = X_valid.drop(columns=all_null_cols)
        X_test = X_test.drop(columns=all_null_cols)

    # 中位数填充
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_valid = X_valid.fillna(medians)
    X_test = X_test.fillna(medians)

    # 提分关键：对 y 做 log 变换，增强对长尾价格分布的拟合稳定性
    model = ExtraTreesRegressor(
        n_estimators=220,
        max_depth=None,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, np.log1p(y_train))

    valid_pred = np.expm1(model.predict(X_valid))
    mae = mean_absolute_error(y_valid, valid_pred)
    r2 = r2_score(y_valid, valid_pred)

    print("===== Improved Baseline 验证集指标 =====")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")
    print(f"数值特征数: {X_train.shape[1]}")

    test_pred = np.expm1(model.predict(X_test))
    out_df = pd.DataFrame({"predicted_price": test_pred})
    out_df.to_csv(OUTPUT_PRED_PATH, index=False)
    print(f"已生成: {OUTPUT_PRED_PATH} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
