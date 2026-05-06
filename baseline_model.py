import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


TRAIN_PATH = "student_train_full.csv"
TEST_PATH = "public_test_questions.csv"
OUTPUT_PRED_PATH = "public_test_predictions.csv"
RANDOM_STATE = 42


def clean_price_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert price like '$150.00' to float and drop missing values."""
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


def main() -> None:
    # 1) 数据读取
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # 2) 基本体检
    print("===== train_df.info() =====")
    train_df.info()
    print("\n===== train_df.describe() =====")
    print(train_df.describe(include="all").transpose().head(30))

    # 3) 价格格式化 + 剔除空价格
    train_df = clean_price_column(train_df)
    print(f"\n清洗后训练集行数: {len(train_df)}")

    # 4) 初级瘦身：删除全空列 + 明显无关列
    full_null_cols = train_df.columns[train_df.isna().all()].tolist()
    drop_irrelevant = [
        "listing_url",
        "id",
        "scrape_id",
        "picture_url",
        "host_url",
        "host_thumbnail_url",
        "host_picture_url",
        "calendar_updated",
    ]
    drop_cols = sorted(set(full_null_cols + drop_irrelevant))

    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors="ignore")

    # AI辅助 baseline 特征选择：选择看起来最直接影响房价的数值特征
    selected_features = [
        "accommodates",
        "bathrooms",
        "bedrooms",
        "beds",
        "latitude",
        "longitude",
        "review_scores_rating",
        "availability_365",
        "number_of_reviews",
        "calculated_host_listings_count",
    ]

    available_features = [f for f in selected_features if f in train_df.columns and f in test_df.columns]
    if len(available_features) < 5:
        raise ValueError(f"可用特征过少，仅有: {available_features}")

    print(f"\n使用特征 ({len(available_features)}): {available_features}")

    # 2) 空值填充：中位数
    X = train_df[available_features].copy()
    y = train_df["price"].copy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_valid = X_valid.fillna(medians)

    # 3) 运行模型
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 4) 指标评估
    valid_pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, valid_pred)
    r2 = r2_score(y_valid, valid_pred)

    print("\n===== 验证集指标 =====")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")

    # 生成 public_test 预测
    test_X = test_df[available_features].copy().fillna(medians)
    test_pred = model.predict(test_X)

    pred_df = pd.DataFrame({"predicted_price": test_pred})
    pred_df.to_csv(OUTPUT_PRED_PATH, index=False)
    print(f"\n已生成预测文件: {OUTPUT_PRED_PATH} (rows={len(pred_df)})")


if __name__ == "__main__":
    main()
