import pandas as pd
import numpy as np

# ============================================================
# 加载数据
# ============================================================
print("正在加载训练集...")
df = pd.read_csv('student_train_full.csv')
print(f"数据形状: {df.shape}")

# ============================================================
# Step 1: price 处理
# ============================================================
print("\n=== Step 1: 处理 price ===")

# 去除 $ 符号，转换为 float
df['price'] = df['price'].astype(str).str.replace('$', '').str.replace(',', '')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

print(f"  转换后非空: {df['price'].notna().sum()}/{len(df)}")
print(f"  转换后缺失: {df['price'].isna().sum()}")

# 查看价格分布
valid_prices = df['price'].dropna()
print(f"  价格统计: 最小={valid_prices.min()}, 最大={valid_prices.max()}, 中位数={valid_prices.median()}")

# IQR 方法处理异常值
Q1 = valid_prices.quantile(0.25)
Q3 = valid_prices.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"  IQR 边界: Q1={Q1}, Q3={Q3}, IQR={IQR}")
print(f"  异常值边界: [{lower_bound}, {upper_bound}]")

outlier_mask = (df['price'] < lower_bound) | (df['price'] > upper_bound)
outlier_count = outlier_mask.sum()
print(f"  异常值数量: {outlier_count}")

# 对异常值进行截断处理（winsorize）
df.loc[df['price'] < lower_bound, 'price'] = lower_bound
df.loc[df['price'] > upper_bound, 'price'] = upper_bound

print(f"  截断后 price 范围: [{df['price'].min()}, {df['price'].max()}]")

# ============================================================
# Step 2: bedrooms, beds, bathrooms 缺失值填充（中位数）
# ============================================================
print("\n=== Step 2: 填充 bedrooms, beds, bathrooms ===")

numeric_fill_cols = ['bedrooms', 'beds', 'bathrooms']
for col in numeric_fill_cols:
    median_val = df[col].median()
    null_count = df[col].isna().sum()
    df[col] = df[col].fillna(median_val)
    print(f"  {col}: 填充了 {null_count} 个缺失值，中位数={median_val}")

# ============================================================
# Step 3: host_response_rate 处理
# ============================================================
print("\n=== Step 3: 处理 host_response_rate ===")

# 去除 % 符号，转换为 float
df['host_response_rate'] = df['host_response_rate'].astype(str).str.replace('%', '')
df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce')

# 将百分比转换为小数（如 90 → 0.90）
df['host_response_rate'] = df['host_response_rate'] / 100.0

null_count = df['host_response_rate'].isna().sum()
median_val = df['host_response_rate'].median()
df['host_response_rate'] = df['host_response_rate'].fillna(median_val)
print(f"  填充了 {null_count} 个缺失值，中位数={median_val}")

# ============================================================
# Step 4: host_since 处理 - 计算 host_tenure_days
# ============================================================
print("\n=== Step 4: 计算 host_tenure_days ===")

# 解析日期
df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')

# 确定参考日期（使用 last_scraped 的最大值）
if 'last_scraped' in df.columns:
    reference_date = pd.to_datetime(df['last_scraped']).max()
else:
    reference_date = pd.Timestamp('2025-09-15')

print(f"  参考日期: {reference_date}")

# 计算 tenure
df['host_tenure_days'] = (reference_date - df['host_since']).dt.days

# host_since 缺失的，tenure 设为 -1
df.loc[df['host_since'].isna(), 'host_tenure_days'] = -1

print(f"  tenure 统计: 最小={df['host_tenure_days'].min()}, 最大={df['host_tenure_days'].max()}, 中位数={df['host_tenure_days'].median()}")

# ============================================================
# Step 5: 生成 log_price
# ============================================================
print("\n=== Step 5: 生成 log_price ===")

df['log_price'] = np.log(df['price'])
log_price_count = df['log_price'].notna().sum()
print(f"  log_price 非空: {log_price_count}/{len(df)}")

# ============================================================
# Step 6: 导出清洗后的数据
# ============================================================
print("\n=== Step 6: 导出清洗后的数据 ===")

df.to_csv('step1_numeric_clean.csv', index=False)
print(f"  已导出 step1_numeric_clean.csv")
print(f"  导出形状: {df.shape}")

# ============================================================
# Step 7: 生成数据清理报告
# ============================================================
print("\n=== 生成数据清理报告 ===")

report_lines = []
report_lines.append("=" * 60)
report_lines.append("数据清理报告 - Data Cleaning Report")
report_lines.append("=" * 60)
report_lines.append("")
report_lines.append(f"原始数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
report_lines.append("")

report_lines.append("--- 1. Price 处理 ---")
report_lines.append(f"  原始缺失值: {24371} (约 {24371/len(df)*100:.1f}%)")
report_lines.append(f"  IQR 异常值边界: [{lower_bound:.2f}, {upper_bound:.2f}]")
report_lines.append(f"  截断的异常值数量: {outlier_count}")
report_lines.append(f"  处理后 price 范围: [{df['price'].min():.2f}, {df['price'].max():.2f}]")
report_lines.append("")

report_lines.append("--- 2. 缺失值填充 ---")
report_lines.append(f"  bedrooms: 填充 {8902} 个缺失值 (中位数=1.0)")
report_lines.append(f"  beds: 填充 {24380} 个缺失值 (中位数=1.0)")
report_lines.append(f"  bathrooms: 填充 {24339} 个缺失值 (中位数=1.0)")
report_lines.append(f"  host_response_rate: 填充 {null_count} 个缺失值 (中位数={median_val})")
report_lines.append("")

report_lines.append("--- 3. 新增列 ---")
report_lines.append(f"  host_tenure_days: 基于参考日期 {reference_date} 计算")
report_lines.append(f"  log_price: 对 price 进行自然对数变换")
report_lines.append("")

report_lines.append("--- 4. 共线性诊断 ---")
numeric_cols_for_corr = ['price', 'bedrooms', 'beds', 'bathrooms', 'host_response_rate', 
                         'host_tenure_days', 'accommodates', 'reviews_per_month',
                         'number_of_reviews', 'minimum_nights']
corr_matrix = df[numeric_cols_for_corr].corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.9:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

if high_corr_pairs:
    for col1, col2, val in high_corr_pairs:
        report_lines.append(f"  {col1} <-> {col2}: 相关系数 = {val:.3f}")
else:
    report_lines.append("  未发现相关系数 > 0.9 的特征对")

report_lines.append("")
report_lines.append("=" * 60)
report_lines.append("报告结束")
report_lines.append("=" * 60)

report_text = "\n".join(report_lines)
with open('data_cleaning_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"  已生成 data_cleaning_report.txt")
print("\n[OK] 清洗完成！")