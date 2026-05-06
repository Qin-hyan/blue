import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy import stats

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建 plots 文件夹
os.makedirs('plots', exist_ok=True)

print("正在加载清洗后的数据...")
df = pd.read_csv('step1_numeric_clean.csv')
print(f"数据形状: {df.shape}")

# ============================================================
# 第一部分：5 个商业问题图表
# ============================================================

# ---------- 问题 1：地段歧视 ----------
print("\n=== 问题 1：地段歧视 ===")

fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(df['longitude'], df['latitude'], 
                     c=df['price'], cmap='hot', alpha=0.3, s=10, edgecolors='none')
ax.scatter(-0.1278, 51.5074, c='white', s=100, marker='*', edgecolors='black', linewidths=2, label='市中心')
ax.set_xlabel('经度 (Longitude)', fontsize=12)
ax.set_ylabel('纬度 (Latitude)', fontsize=12)
ax.set_title('伦敦房价地理分布图\n（颜色越红越贵）', fontsize=14)
ax.legend(loc='upper right')
plt.colorbar(scatter, label='价格 (GBP/晚)')
plt.tight_layout()
plt.savefig('plots/1_location_price.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: plots/1_location_price.png")

# ---------- 问题 2：懒人房东 ----------
print("\n=== 问题 2：懒人房东 ===")

df['response_rate_group'] = pd.cut(df['host_response_rate'], 
                                    bins=[-0.01, 0, 0.5, 0.8, 0.95, 1.01],
                                    labels=['缺失/0%', '0-50%', '50-80%', '80-95%', '95-100%'])

fig, ax = plt.subplots(figsize=(10, 6))
groups = df.dropna(subset=['response_rate_group'])
boxes = ax.boxplot([groups[groups['response_rate_group'] == g]['price'].dropna() 
                     for g in groups['response_rate_group'].unique()],
                    labels=[str(g) for g in groups['response_rate_group'].unique()],
                    patch_artist=True, medianprops=dict(color='red', linewidth=2))
for box in boxes['boxes']:
    box.set_facecolor('lightblue')
ax.set_xlabel('回复率分组', fontsize=12)
ax.set_ylabel('价格 (GBP/晚)', fontsize=12)
ax.set_title('不同回复率房东的房价分布', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/2_response_rate_box.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: plots/2_response_rate_box.png")

# ---------- 问题 3：好评的代价 ----------
print("\n=== 问题 3：好评的代价 ===")

valid_review = df.dropna(subset=['review_scores_rating'])

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(valid_review['review_scores_rating'], valid_review['price'], 
           alpha=0.3, s=10, edgecolors='none')

# 趋势线
z = np.polyfit(valid_review['review_scores_rating'], valid_review['price'], 1)
p = np.poly1d(z)
x_line = np.linspace(valid_review['review_scores_rating'].min(), 
                     valid_review['review_scores_rating'].max(), 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'趋势线 (斜率={z[0]:.2f})')

ax.set_xlabel('总体评分 (review_scores_rating)', fontsize=12)
ax.set_ylabel('价格 (GBP/晚)', fontsize=12)
ax.set_title('评分与价格的关系', fontsize=14)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('plots/3_review_score_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: plots/3_review_score_scatter.png")

# ---------- 问题 4：差旅还是度假 ----------
print("\n=== 问题 4：差旅还是度假 ===")

fig, ax = plt.subplots(figsize=(10, 6))
bedrooms_counts = df['bedrooms'].value_counts().sort_index()
bars = ax.bar(bedrooms_counts.index.astype(str), bedrooms_counts.values, 
              color=['lightcoral' if i == bedrooms_counts.values.argmax() else 'skyblue' 
                     for i in range(len(bedrooms_counts))])
ax.set_xlabel('卧室数量', fontsize=12)
ax.set_ylabel('房源数量', fontsize=12)
ax.set_title('伦敦 Airbnb 卧室数量分布', fontsize=14)

# 添加数量标签
for bar, val in zip(bars, bedrooms_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
            str(val), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('plots/4_bedrooms_hist.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: plots/4_bedrooms_hist.png")

# ---------- 问题 5：专业房东 vs 业余房东 ----------
print("\n=== 问题 5：专业房东 vs 业余房东 ===")

# 限制 host_listings_count 范围以便可视化
df_subset = df[df['host_listings_count'] <= 100].copy()

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df_subset['host_listings_count'], df_subset['price'], 
           alpha=0.3, s=10, edgecolors='none')

# 按房源数量分组统计
grouped = df_subset.groupby('host_listings_count')['price'].agg(['mean', 'count'])
ax.plot(grouped.index, grouped['mean'], 'r-', linewidth=2, marker='o', markersize=4, 
        label='平均价格趋势')

ax.set_xlabel('房东管理房源数量', fontsize=12)
ax.set_ylabel('价格 (GBP/晚)', fontsize=12)
ax.set_title('房东经验与房价的关系', fontsize=14)
ax.set_xscale('log')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('plots/5_host_listings_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: plots/5_host_listings_scatter.png")

# ============================================================
# 第二部分：6 个自创问题
# ============================================================

# ---------- 问题 6：房型与价格 ----------
print("\n=== 问题 6：房型与价格 ===")

fig, ax = plt.subplots(figsize=(10, 6))
room_type_stats = df.groupby('room_type')['price'].agg(['mean', 'count']).sort_values('mean', ascending=False)
bars = ax.bar(range(len(room_type_stats)), room_type_stats['mean'].values, 
              color=['lightgreen' if i == room_type_stats['mean'].idxmax() else 'lightyellow' 
                     for i in range(len(room_type_stats))])
ax.set_xticks(range(len(room_type_stats)))
ax.set_xticklabels(room_type_stats.index, rotation=45, ha='right')
ax.set_xlabel('房型 (room_type)', fontsize=12)
ax.set_ylabel('平均价格 (GBP/晚)', fontsize=12)
ax.set_title('不同房型的平均价格', fontsize=14)

for bar, val in zip(bars, room_type_stats['mean'].values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{val:.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('plots/6_room_type_price.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: plots/6_room_type_price.png")

# ---------- 问题 7：最短入住天数与价格 ----------
print("\n=== 问题 7：最短入住天数与价格 ===")

fig, ax = plt.subplots(figsize=(10, 6))
min_nights_stats = df.groupby('minimum_nights')['price'].agg(['mean', 'count']).sort_index()
ax.bar(min_nights_stats.index, min_nights_stats['mean'].values, color='coral')
ax.set_xlabel('最短入住天数 (minimum_nights)', fontsize=12)
ax.set_ylabel('平均价格 (GBP/晚)', fontsize=12)
ax.set_title('最短入住天数与平均价格的关系', fontsize=14)
plt.tight_layout()
plt.savefig('plots/7_minimum_nights_price.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: plots/7_minimum_nights_price.png")

# ---------- 问题 8：超级房东与价格 ----------
print("\n=== 问题 8：超级房东与价格 ===")

fig, ax = plt.subplots(figsize=(8, 6))
superhost_stats = df.groupby('host_is_superhost')['price'].agg(['mean', 'count'])
bars = ax.bar(['非超级房东', '超级房东'], 
              [superhost_stats.loc[False, 'mean'], superhost_stats.loc[True, 'mean']],
              color=['lightblue', 'gold'])
ax.set_ylabel('平均价格 (GBP/晚)', fontsize=12)
ax.set_title('超级房东 vs 普通房东：价格对比', fontsize=14)

for bar, val in zip(bars, [superhost_stats.loc[False, 'mean'], superhost_stats.loc[True, 'mean']]):
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
              [verified_stats.loc[False, 'mean'], verified_stats.loc[True, 'mean']],
              color=['lightgray', 'lightgreen'])
ax.set_ylabel('平均价格 (GBP/晚)', fontsize=12)
ax.set_title('房东身份验证状态与价格的关系', fontsize=14)

for bar, val in zip(bars, [verified_stats.loc[False, 'mean'], verified_stats.loc[True, 'mean']]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/9_identity_verified_price.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: plots/9_identity_verified_price.png")

# ---------- 问题 10：描述长度与价格 ----------
print("\n=== 问题 10：描述长度与价格 ===")

df['name_length'] = df['name'].str.len()
df['desc_length'] = df['description'].str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 名称长度 vs 价格
ax1 = axes[0]
valid_name = df.dropna(subset=['name_length'])
ax1.scatter(valid_name['name_length'], valid_name['price'], alpha=0.3, s=10, edgecolors='none')
z1 = np.polyfit(valid_name['name_length'], valid_name['price'], 1)
p1 = np.poly1d(z1)
x1 = np.linspace(valid_name['name_length'].min(), valid_name['name_length'].max(), 100)
ax1.plot(x1, p1(x1), "r--", linewidth=2)
ax1.set_xlabel('名称长度 (字符数)', fontsize=11)
ax1.set_ylabel('价格 (GBP/晚)', fontsize=11)
ax1.set_title('名称长度与价格的关系')

# 描述长度 vs 价格
ax2 = axes[1]
valid_desc = df.dropna(subset=['desc_length'])
ax2.scatter(valid_desc['desc_length'], valid_desc['price'], alpha=0.3, s=10, edgecolors='none')
z2 = np.polyfit(valid_desc['desc_length'], valid_desc['price'], 1)
p2 = np.poly1d(z2)
x2 = np.linspace(valid_desc['desc_length'].min(), valid_desc['desc_length'].max(), 100)
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

df['amenities_count'] = df['amenities'].apply(lambda x: len(x.split('","')) - 1 if isinstance(x, str) and x != '[]' else 0)

fig, ax = plt.subplots(figsize=(10, 6))
valid_amen = df.dropna(subset=['amenities_count'])
ax.scatter(valid_amen['amenities_count'], valid_amen['price'], alpha=0.3, s=10, edgecolors='none')

# 趋势线
z3 = np.polyfit(valid_amen['amenities_count'], valid_amen['price'], 1)
p3 = np.poly1d(z3)
x3 = np.linspace(valid_amen['amenities_count'].min(), valid_amen['amenities_count'].max(), 100)
ax.plot(x3, p3(x3), "r--", linewidth=2, label=f'趋势线 (斜率={z3[0]:.2f})')

ax.set_xlabel('设施数量 (amenities)', fontsize=12)
ax.set_ylabel('价格 (GBP/晚)', fontsize=12)
ax.set_title('设施数量与价格的关系', fontsize=14)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('plots/11_amenities_count_price.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: plots/11_amenities_count_price.png")

# ============================================================
# 第三部分：数据清理操作
# ============================================================

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
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
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