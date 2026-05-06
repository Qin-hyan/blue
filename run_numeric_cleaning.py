import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def main() -> None:
    df = pd.read_csv("student_train_full.csv")
    df_work = df.copy()

    numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
    df_num = df_work[numeric_cols].copy()

    # 仅对训练集价格做清洗，避免 price 为空影响后续统计
    if "price" in df_work.columns:
        price_clean = (
            df_work["price"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        price_clean = price_clean.replace({"": np.nan, "nan": np.nan, "None": np.nan})
        price_clean = pd.to_numeric(price_clean, errors="coerce")
        df_num["price"] = price_clean

    # 删除全空数值列，避免中位数填充后仍为 NaN
    all_null_numeric_cols = [c for c in df_num.columns if df_num[c].isna().all()]
    if all_null_numeric_cols:
        df_num = df_num.drop(columns=all_null_numeric_cols)

    skew_before = df_num.skew(numeric_only=True)
    skew_before.sort_values(ascending=False).to_csv("skew_before.csv", header=["skew_before"])

    # 全量数值列直方图（保存图）
    n_cols = 3
    n_rows = int(np.ceil(len(df_num.columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.8 * n_rows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(df_num.columns):
        axes[i].hist(df_num[col].dropna(), bins=40, alpha=0.75)
        axes[i].set_title(f"{col}\nskew={df_num[col].skew():.2f}")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig("numeric_hist_before.png", dpi=150)
    plt.close(fig)

    # 业务异常修正
    if "bedrooms" in df_num.columns:
        valid_mask = (df_num["bedrooms"] > 0) & (df_num["bedrooms"] <= 10)
        med_bedrooms = df_num.loc[valid_mask, "bedrooms"].median()
        bad_mask = (df_num["bedrooms"] == 0) | (df_num["bedrooms"] > 10)
        df_num.loc[bad_mask, "bedrooms"] = med_bedrooms

    if "area" in df_num.columns:
        df_num["area"] = df_num["area"].clip(lower=100, upper=5000)

    # 通用 IQR 截断
    for col in df_num.columns:
        s = df_num[col]
        if s.dropna().shape[0] < 5:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df_num[col] = s.clip(lower=lower, upper=upper)

    # 缺失值中位数填充（标记）
    na_before = df_num.isna().sum()
    cols_with_na = na_before[na_before > 0].index.tolist()
    pd.Series(cols_with_na, name="filled_by_median").to_csv("filled_na_columns.csv", index=False)
    for col in cols_with_na:
        df_num[col] = df_num[col].fillna(df_num[col].median())

    # 右偏列 log1p
    skew_mid = df_num.skew(numeric_only=True)
    right_skew_cols = [c for c in df_num.columns if (skew_mid[c] > 1.0 and df_num[c].min() >= 0)]
    for must_col in ["price", "Price", "area", "Area"]:
        if must_col in df_num.columns and df_num[must_col].min() >= 0 and must_col not in right_skew_cols:
            right_skew_cols.append(must_col)

    pd.Series(right_skew_cols, name="log1p_columns").to_csv("log1p_columns.csv", index=False)

    for col in right_skew_cols:
        before = df_num[col].copy()
        df_num[col] = np.log1p(df_num[col])

        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        ax[0].hist(before.dropna(), bins=40, alpha=0.75, color="steelblue")
        ax[0].set_title(f"{col} - Before\nskew={before.skew():.2f}")
        ax[1].hist(df_num[col].dropna(), bins=40, alpha=0.75, color="seagreen")
        ax[1].set_title(f"{col} - After log1p\nskew={df_num[col].skew():.2f}")
        plt.tight_layout()
        plt.savefig(f"log1p_compare_{col}.png", dpi=150)
        plt.close(fig)

    skew_after = df_num.skew(numeric_only=True)
    skew_after.sort_values(ascending=False).to_csv("skew_after.csv", header=["skew_after"])

    # 标准化
    scaler = StandardScaler()
    df_numeric_clean = pd.DataFrame(
        scaler.fit_transform(df_num),
        columns=df_num.columns,
        index=df_num.index,
    )

    df_numeric_clean.to_csv("df_numeric_clean.csv", index=False)

    # 总结输出
    print("=== Numeric Cleaning Completed ===")
    print(f"Input shape: {df.shape}")
    print(f"Numeric columns: {len(df_num.columns)}")
    print(f"Dropped all-null numeric columns: {len(all_null_numeric_cols)}")
    print(f"Filled NA columns: {len(cols_with_na)}")
    print(f"log1p columns: {len(right_skew_cols)}")
    print(f"Output file: df_numeric_clean.csv, shape={df_numeric_clean.shape}")
    print("Saved: skew_before.csv, skew_after.csv, filled_na_columns.csv, log1p_columns.csv")
    print("Saved plots: numeric_hist_before.png and log1p_compare_*.png")


if __name__ == "__main__":
    main()
