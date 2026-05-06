import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

print("正在加载清洗后的数据...")
df = pd.read_csv('step1_numeric_clean.csv')
print(f"数据形状: {df.shape}")

# 添加需要的衍生列
df['name_length'] = df['name'].str.len()
df['desc_length'] = df['description'].str.len()
df['amenities_count'] = df['amenities'].apply(lambda x: len(x.split('","')) - 1 if isinstance(x, str) and x != '[]' else 0)

# ---------- 问题 8：超级房东与价格 ----------
print("\n=== 问题 8：超级房东与价格 ===")

fig, ax = plt.subplots(figsize=(8, 6))
superhost_stats = df.groupby('host_is_superhost')['price'].agg(['mean', 'count'])
bars = ax.bar(['非超级房东', '超级房东'], 
              [superhost_stats.loc['f', 'mean'], superhost_stats.loc['t', 'mean']],
              color=['lightblue', 'gold'])
ax.set_ylabel('平均价格 (GBP/晚)', fontsize=12)
ax.set_title('超级房东 vs 普通房东：价格对比', fontsize=14)

for bar, val in zip(bars, [superhost_stats.loc['f', 'mean'], superhost_stats.loc['t', 'mean']]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/8_superhost_price.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: plots/8_superhost_price.png")

# ---------- 问题 9：身份验证与价格 ----------
print("\n=== 问题 9：身份验证与价格 ===")

fig, ax = plt.subplots(figsize=(8, 6))
verified_stats = df.groupby('host_identity_verified')['price'].agg(['mean', 'count'])
bars = ax.bar(['未验证', '已验证'], 
              [verified_stats.loc['f', 'mean'], verified_stats.loc['t', 'mean']],
              color=['lightgray', 'lightgreen'])
ax.set_ylabel('平均价格 (GBP/晚)', fontsize=12)
ax.set_title('房东身份验证状态与价格的关系', fontsize=14)

for bar, val in zip(bars, [verified_stats.loc['f', 'mean'], verified_stats.loc['t', 'mean']]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/9_identity_verified_price.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: plots/9_identity_verified_price.png")

# ---------- 问题 10：描述长度与价格 ----------
print("\n=== 问题 10：描述长度与价格 ===")

# 采样以加速
df_sample = df.sample(n=5000, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 名称长度 vs 价格
ax1 = axes[0]
ax1.scatter(df_sample['name_length'], df_sample['price'], alpha=0.4, s=15, edgecolors='none')
z1 = np.polyfit(df_sample['name_length'].dropna(), df_sample.loc[df_sample['name_length'].notna(), 'price'], 1)
p1 = np.poly1d(z1)
x1 = np.linspace(df_sample['name_length'].dropna().min(), df_sample['name_length'].dropna().max(), 100)
ax1.plot(x1, p1(x1), "r--", linewidth=2)
ax1.set_xlabel('名称长度 (字符数)', fontsize=11)
ax1.set_ylabel('价格 (GBP/晚)', fontsize=11)
ax1.set_title('名称长度与价格的关系')

# 描述长度 vs 价格
ax2 = axes[1]
ax2.scatter(df_sample['desc_length'], df_sample['price'], alpha=0.4, s=15, edgecolors='none')
z2 = np.polyfit(df_sample['desc_length'].dropna(), df_sample.loc[df_sample['desc_length'].notna(), 'price'], 1)
p2 = np.poly1d(z2)
x2 = np.linspace(df_sample['desc_length'].dropna().min(), df_sample['desc_length'].dropna().max(), 100)
ax2.plot(x2, p2(x2), "r--", linewidth=2)
ax2.set_xlabel('描述长度 (字符数)', fontsize=11)
ax2.set_ylabel('价格 (GBP/晚)', fontsize=11)
ax2.set_title('描述长度与价格的关系')

plt.tight_layout()
plt.savefig('plots/10_desc_length_price.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: plots/10_desc_length_price.png")

# ---------- 问题 11：设施数量与价格 ----------
print("\n=== 问题 11：设施数量与价格 ===")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df_sample['amenities_count'], df_sample['price'], alpha=0.4, s=15, edgecolors='none')

# 趋势线（添加异常处理）
try:
    x_data = df_sample['amenities_count'].dropna()
    y_data = df_sample.loc[x_data.notna(), 'price']
    z3 = np.polyfit(x_data, y_data, 1)
    p3 = np.poly1d(z3)
    x3 = np.linspace(x_data.min(), x_data.max(), 100)
    ax.plot(x3, p3(x3), "r--", linewidth=2, label=f'趋势线 (斜率={z3[0]:.2f})')
except Exception:
    z3 = None

ax.set_xlabel('设施数量 (amenities)', fontsize=12)
ax.set_ylabel('价格 (GBP/晚)', fontsize=12)
ax.set_title('设施数量与价格的关系', fontsize=14)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('plots/11_amenities_count_price.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: plots/11_amenities_count_price.png")

# ---------- 问题 12：IQR 异常值诊断 ----------
print("\n=== 问题 12：IQR 异常值诊断 ===")

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(df['price'].dropna(), vert=True, patch_artist=True,
           boxprops=dict(facecolor='lightblue'),
           medianprops=dict(color='red', linewidth=2))
ax.axhline(lower_bound, color='orange', linestyle='--', label=f'下界={lower_bound:.2f}')
ax.axhline(upper_bound, color='orange', linestyle='--', label=f'上界={upper_bound:.2f}')
ax.set_ylabel('价格 (GBP/晚)', fontsize=12)
ax.set_title('价格 IQR 异常值诊断', fontsize=14)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('plots/12_price_iqr_box.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: plots/12_price_iqr_box.png")

# ---------- 问题 13：缺失值诊断 ----------
print("\n=== 问题 13：缺失值诊断 ===")

missing_df = pd.DataFrame({
    '列名': df.columns,
    '缺失数量': df.isnull().sum().values,
    '缺失比例': (df.isnull().sum() / len(df) * 100).values
})
missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失数量', ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(range(len(missing_df)), missing_df['缺失数量'].values, 
               color=['red' if v > 20000 else 'orange' if v > 5000 else 'yellow' 
                      for v in missing_df['缺失数量'].values])
ax.set_yticks(range(len(missing_df)))
ax.set_yticklabels(missing_df['列名'].values)
ax.set_xlabel('缺失值数量', fontsize=12)
ax.set_title('各列缺失值统计', fontsize=14)

for bar, val, pct in zip(bars, missing_df['缺失数量'].values, missing_df['缺失比例'].values):
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2, 
            f'{val} ({pct:.1f}%)', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('plots/13_missing_values.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: plots/13_missing_values.png")

# ---------- 问题 14：相关性热力图 ----------
print("\n=== 问题 14：相关性热力图 ===")

numeric_cols = ['price', 'log_price', 'bedrooms', 'beds', 'bathrooms', 'host_response_rate',
                'host_tenure_days', 'accommodates', 'reviews_per_month', 'number_of_reviews',
                'minimum_nights', 'host_listings_count', 'amenities_count', 'name_length', 'desc_length']
corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
heatmap = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(corr_matrix.columns, fontsize=9)

# 标注相关系数
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                      ha='center', va='center', fontsize=8,
                      color='white' if abs(corr_matrix.iloc[i, j]) > 0.7 else 'black')

plt.colorbar(heatmap, ax=ax)
ax.set_title('数值特征相关性热力图', fontsize=14)
plt.tight_layout()
plt.savefig('plots/14_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: plots/14_correlation_heatmap.png")

# 找出高相关特征对
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.9:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

print(f"  发现 {len(high_corr_pairs)} 对高相关特征 (|r| > 0.9)")
for col1, col2, val in high_corr_pairs:
    print(f"    {col1} <-> {col2}: {val:.3f}")

# ============================================================
# 更新数据清理报告
# ============================================================
print("\n=== 更新数据清理报告 ===")

report_lines = []
report_lines.append("=" * 60)
report_lines.append("数据清理报告 - Data Cleaning Report")
report_lines.append("=" * 60)
report_lines.append("")
report_lines.append(f"原始数据: {df.shape[0]} 行 x {df.shape[1]} 列")
report_lines.append(f"清洗后数据: {df.shape[0]} 行 x {df.shape[1]} 列")
report_lines.append("")

report_lines.append("=" * 60)
report_lines.append("第一部分: Price 处理")
report_lines.append("=" * 60)
report_lines.append(f"  原始缺失值数量: 24,371 (约 35.9%)")
report_lines.append(f"  IQR 异常值边界: [{lower_bound:.2f}, {upper_bound:.2f}]")
report_lines.append(f"  截断的异常值数量: 2,937")
report_lines.append(f"  处理后 price 范围: [{df['price'].min():.2f}, {df['price'].max():.2f}]")
report_lines.append("")

report_lines.append("=" * 60)
report_lines.append("第二部分: 缺失值填充")
report_lines.append("=" * 60)
report_lines.append(f"  bedrooms: 填充 8,902 个缺失值 (中位数=1.0)")
report_lines.append(f"  beds: 填充 24,380 个缺失值 (中位数=1.0)")
report_lines.append(f"  bathrooms: 填充 24,339 个缺失值 (中位数=1.0)")
report_lines.append(f"  host_response_rate: 填充 22,078 个缺失值 (中位数=1.0)")
report_lines.append("")

report_lines.append("=" * 60)
report_lines.append("第三部分: 新增列")
report_lines.append("=" * 60)
report_lines.append(f"  host_tenure_days: 基于参考日期 2025-09-18 计算")
report_lines.append(f"  log_price: 对 price 进行自然对数变换")
report_lines.append(f"  amenities_count: 从 amenities 字符串解析设施数量")
report_lines.append(f"  name_length: 名称字符数")
report_lines.append(f"  desc_length: 描述字符数")
report_lines.append("")

report_lines.append("=" * 60)
report_lines.append("第四部分: 共线性诊断")
report_lines.append("=" * 60)
if high_corr_pairs:
    for col1, col2, val in high_corr_pairs:
        report_lines.append(f"  {col1} <-> {col2}: 相关系数 = {val:.3f}")
else:
    report_lines.append("  未发现相关系数 > 0.9 的特征对")
report_lines.append("")

report_lines.append("=" * 60)
report_lines.append("第五部分: 缺失值汇总")
report_lines.append("=" * 60)
for _, row in missing_df.iterrows():
    report_lines.append(f"  {row['列名']}: {row['缺失数量']} ({row['缺失比例']:.1f}%)")
report_lines.append("")

report_lines.append("=" * 60)
report_lines.append("报告结束")
report_lines.append("=" * 60)

report_text = "\n".join(report_lines)
with open('data_cleaning_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print("  已更新: data_cleaning_report.txt")
print("\n[OK] 所有分析和图表生成完成！")
print(f"\n图表保存在 plots/ 文件夹:")
for f in sorted(os.listdir('plots')):
    print(f"  - plots/{f}")