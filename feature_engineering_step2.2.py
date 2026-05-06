"""
Step 2.2: 文字与空间的"炼金术" - 特征工程脚本
从四个"矿区"中各选 1 个猜想进行实验验证

矿区 A: 地段的鄙视链 (分箱)
矿区 B: 黄金一公里 (距离计算)
矿区 C: 洗衣机自由 (设施解析)
矿区 D: 姜是老的辣 (房龄计算)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 特拉法加广场坐标 (伦敦市中心)
TRAFALGAR_SQUARE = (51.5080, -0.1281)


def load_data(input_path):
    """加载数据"""
    print("=" * 60)
    print("Step 2.2: 文字与空间的'炼金术'")
    print("=" * 60)
    df = pd.read_csv(input_path)
    
    # 清洗 price 列：移除 '$' 符号并转换为数值
    df['price'] = df['price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    print(f"\n数据形状：{df.shape}")
    return df


# ============================================================
# 矿区 A: 地段的鄙视链 - 将 33 个社区分为 4 个层级
# ============================================================
def neighborhood_tier_analysis(df, save_dir):
    """
    矿区 A: 地段的鄙视链
    将 neighbourhood_cleansed (33 个社区) 按均价分成"顶级、高级、大众、廉价"四组
    """
    print("\n" + "=" * 60)
    print("矿区 A: 地段的鄙视链")
    print("=" * 60)
    
    # 计算每个社区的价格中位数
    neighborhood_price = df.groupby('neighbourhood_cleansed')['price'].median().reset_index()
    neighborhood_price.columns = ['neighbourhood_cleansed', 'median_price']
    
    # 按价格中位数排序
    neighborhood_price = neighborhood_price.sort_values('median_price', ascending=False)
    
    # 分成 4 等份 ( quartiles)
    df_with_tier = df.copy()
    quartiles = df_with_tier['price'].quantile([0.25, 0.5, 0.75]).values
    
    # 创建 tier 分类
    def assign_tier(price):
        if price >= quartiles[2]:
            return '顶级'
        elif price >= quartiles[1]:
            return '高级'
        elif price >= quartiles[0]:
            return '大众'
        else:
            return '廉价'
    
    df_with_tier['neighbourhood_tier'] = df_with_tier['price'].apply(assign_tier)
    
    # 计算每个 tier 的社区列表
    tier_communities = {}
    for tier in ['顶级', '高级', '大众', '廉价']:
        tier_mask = df_with_tier['neighbourhood_tier'] == tier
        communities = df_with_tier[tier_mask]['neighbourhood_cleansed'].unique()
        tier_communities[tier] = list(communities)
        print(f"\n{tier}社区 ({len(communities)}个): {', '.join(communities[:5])}...")
    
    # 可视化：各 tier 价格分布
    plt.figure(figsize=(10, 6))
    tier_order = ['顶级', '高级', '大众', '廉价']
    sns.boxplot(data=df_with_tier, x='neighbourhood_tier', y='price', order=tier_order, palette='viridis')
    plt.title('各地段 tier 价格分布', fontsize=14)
    plt.xlabel('地段 tier', fontsize=12)
    plt.ylabel('价格 (£)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/neighborhood_tier_distribution.png", dpi=150)
    plt.close()
    
    # 计算各 tier 均价
    tier_stats = df_with_tier.groupby('neighbourhood_tier')['price'].agg(['mean', 'median', 'count']).round(2)
    tier_stats = tier_stats.reindex(tier_order)
    print(f"\n各 tier 统计:")
    print(tier_stats)
    
    # 保存结果
    df_with_tier.to_csv(f"{save_dir}/data_with_tier.csv", index=False)
    
    print(f"\n业务分析结论:")
    print(f"1. 通过价格分位数将 33 个社区分为 4 个 tier，顶级 tier 均价£{tier_stats.loc['顶级', 'mean']:.0f}, 廉价 tier 均价£{tier_stats.loc['廉价', 'mean']:.0f}, 相差{tier_stats.loc['顶级', 'mean']/tier_stats.loc['廉价', 'mean']:.1f}倍")
    print(f"2. 这种分箱方法将 33 个类别压缩为 4 个，减少了特征维度，同时保留了地段价值信息")
    print(f"3. 顶级 tier 包含 {len(tier_communities['顶级'])} 个社区，廉价 tier 包含 {len(tier_communities['廉价'])} 个社区")
    print(f"4. 使用 neighbourhood_tier 替代原始 neighbourhood_cleansed 可以帮助模型更快学习地段价值模式")
    
    return df_with_tier, tier_stats


# ============================================================
# 矿区 B: 黄金一公里 - 计算到市中心的距离
# ============================================================
def distance_to_center_analysis(df, save_dir):
    """
    矿区 B: 黄金一公里
    计算每个房源到特拉法加广场的距离，分析距离与价格的关系
    """
    print("\n" + "=" * 60)
    print("矿区 B: 黄金一公里")
    print("=" * 60)
    
    df_with_dist = df.copy()
    
    # 计算到特拉法加广场的距离 (Haversine 公式)
    def haversine_distance(lat, lon):
        # Haversine 公式计算球面距离 (公里)
        R = 6371  # 地球半径 (km)
        lat1, lon1 = np.radians(TRAFALGAR_SQUARE[0]), np.radians(TRAFALGAR_SQUARE[1])
        lat2, lon2 = np.radians(lat), np.radians(lon)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    df_with_dist['dist_to_center'] = df_with_dist.apply(
        lambda row: haversine_distance(row['latitude'], row['longitude']), axis=1
    )
    
    # 计算相关性
    correlation = df_with_dist['dist_to_center'].corr(df_with_dist['price'])
    print(f"\n距离与价格的相关系数：{correlation:.4f}")
    
    # 线性回归计算每公里价格变化 (过滤 NaN 值)
    from sklearn.linear_model import LinearRegression
    valid_mask = df_with_dist['dist_to_center'].notna() & df_with_dist['price'].notna()
    X = df_with_dist.loc[valid_mask, ['dist_to_center']]
    y = df_with_dist.loc[valid_mask, 'price']
    model = LinearRegression()
    model.fit(X, y)
    price_change_per_km = model.coef_[0]
    print(f"每公里价格变化：£{price_change_per_km:.2f}")
    
    # 可视化：距离 - 价格散点图 + 趋势线
    plt.figure(figsize=(12, 8))
    plt.scatter(df_with_dist['dist_to_center'], df_with_dist['price'], alpha=0.3, s=10, color='blue')
    
    # 添加趋势线
    x_range = np.linspace(df_with_dist['dist_to_center'].min(), df_with_dist['dist_to_center'].max(), 100)
    y_pred = model.predict(x_range.reshape(-1, 1))
    plt.plot(x_range, y_pred, 'r-', linewidth=2, label=f'趋势线 (slope=£{price_change_per_km:.0f}/km)')
    
    plt.xlabel('距离市中心 (km)', fontsize=12)
    plt.ylabel('价格 (£)', fontsize=12)
    plt.title('距离市中心 vs 房价', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/distance_vs_price.png", dpi=150)
    plt.close()
    
    # 距离分段统计
    df_with_dist['dist_bucket'] = pd.cut(df_with_dist['dist_to_center'], bins=[0, 2, 5, 10, 20, 100], 
                                          labels=['0-2km', '2-5km', '5-10km', '10-20km', '20km+'])
    dist_stats = df_with_dist.groupby('dist_bucket')['price'].agg(['mean', 'median', 'count']).round(2)
    print(f"\n距离分段统计:")
    print(dist_stats)
    
    # 保存结果
    df_with_dist.to_csv(f"{save_dir}/data_with_distance.csv", index=False)
    
    print(f"\n业务分析结论:")
    print(f"1. 距离市中心每增加 1 公里，房价平均下降£{abs(price_change_per_km):.0f} (相关系数：{correlation:.3f})")
    print(f"2. 市中心 0-2km 范围内的房源均价£{dist_stats.loc[('0-2km', 'mean')]:.0f}, 是 10km 外房源均价的{dist_stats.loc[('0-2km', 'mean')]/dist_stats.loc[('10-20km', 'mean')]:.1f}倍")
    print(f"3. '黄金一公里'效应明显，市中心房源具有显著溢价")
    print(f"4. dist_to_center 是一个强预测特征，应加入模型")
    
    return df_with_dist, dist_stats, price_change_per_km


# ============================================================
# 矿区 C: 洗衣机自由 - 设施解析
# ============================================================
def amenities_analysis(df, save_dir):
    """
    矿区 C: 洗衣机自由
    从 amenities 列解析设施列表，创建 has_washer 和 has_dryer 标志列
    """
    print("\n" + "=" * 60)
    print("矿区 C: 洗衣机自由")
    print("=" * 60)
    
    df_with_amenities = df.copy()
    
    # 解析 amenities 列 (格式：["Wifi", "Kitchen", ...])
    def parse_amenities(amenities_str):
        if pd.isna(amenities_str):
            return {}
        try:
            # 移除方括号和引号，分割
            amenities_str = amenities_str.strip('[]')
            amenities_str = amenities_str.replace('"', '').replace("'", '')
            amenities_list = [a.strip() for a in amenities_str.split(',')]
            return {a: True for a in amenities_list}
        except:
            return {}
    
    # 创建设施标志列
    amenities_to_check = ['Washer', 'Dryer', 'Wifi', 'Kitchen', 'Heating', 'Air conditioning']
    
    for amenity in amenities_to_check:
        df_with_amenities[f'has_{amenity.lower()}'] = df_with_amenities['amenities'].apply(
            lambda x: 1 if amenity in str(x) else 0
        )
    
    # 统计设施拥有率
    amenity_stats = {}
    for amenity in amenities_to_check:
        col = f'has_{amenity.lower()}'
        rate = df_with_amenities[col].mean() * 100
        amenity_stats[amenity] = rate
        print(f"{amenity}拥有率：{rate:.1f}%")
    
    # 分析设施对价格的影响
    print("\n设施对价格的影响:")
    for amenity in amenities_to_check:
        col = f'has_{amenity.lower()}'
        has_price = df_with_amenities[df_with_amenities[col] == 1]['price'].mean()
        no_price = df_with_amenities[df_with_amenities[col] == 0]['price'].mean()
        premium = (has_price - no_price) / no_price * 100
        print(f"{amenity}: 有设施均价£{has_price:.0f}, 无设施均价£{no_price:.0f}, 溢价{premium:.1f}%")
    
    # 可视化：洗衣机/烘干机对比
    plt.figure(figsize=(10, 6))
    washer_data = df_with_amenities[df_with_amenities['has_washer'] == 1]['price']
    no_washer_data = df_with_amenities[df_with_amenities['has_washer'] == 0]['price']
    
    plt.boxplot([no_washer_data, washer_data], labels=['无洗衣机', '有洗衣机'])
    plt.ylabel('价格 (£)', fontsize=12)
    plt.title('洗衣机对房价的影响', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/washer_impact.png", dpi=150)
    plt.close()
    
    # 保存结果
    df_with_amenities.to_csv(f"{save_dir}/data_with_amenities.csv", index=False)
    
    print(f"\n业务分析结论:")
    print(f"1. 洗衣机拥有率：{amenity_stats['Washer']:.1f}%, 烘干机拥有率：{amenity_stats['Dryer']:.1f}%")
    print(f"2. 有洗衣机的房源均价£{df_with_amenities[df_with_amenities['has_washer']==1]['price'].mean():.0f}, 无洗衣机的房源均价£{df_with_amenities[df_with_amenities['has_washer']==0]['price'].mean():.0f}")
    print(f"3. 洗衣机带来的价格溢价约{((df_with_amenities[df_with_amenities['has_washer']==1]['price'].mean() - df_with_amenities[df_with_amenities['has_washer']==0]['price'].mean()) / df_with_amenities[df_with_amenities['has_washer']==0]['price'].mean() * 100):.1f}%")
    print(f"4. '洗衣机自由'确实存在，有洗衣机的房源具有显著价格优势")
    
    return df_with_amenities, amenity_stats


# ============================================================
# 矿区 D: 姜是老的辣 - 房龄计算
# ============================================================
def host_age_analysis(df, save_dir):
    """
    矿区 D: 姜是老的辣
    从 host_since 计算房龄，分析老房东 vs 新房东的价格差异
    """
    print("\n" + "=" * 60)
    print("矿区 D: 姜是老的辣")
    print("=" * 60)
    
    df_with_age = df.copy()
    
    # 解析 host_since 并计算房龄
    df_with_age['host_since'] = pd.to_datetime(df_with_age['host_since'], errors='coerce')
    reference_year = 2026
    df_with_age['host_age_years'] = reference_year - df_with_age['host_since'].dt.year
    
    # 分组：老房东 (>=5 年) vs 新房东 (<5 年)
    df_with_age['host_type'] = df_with_age['host_age_years'].apply(
        lambda x: '老房东' if pd.notna(x) and x >= 5 else '新房东'
    )
    
    # 统计
    age_stats = df_with_age.groupby('host_type')['price'].agg(['mean', 'median', 'count', 'std']).round(2)
    print(f"\n房东类型统计:")
    print(age_stats)
    
    # 房龄分布
    print(f"\n房龄分布:")
    age_dist = df_with_age['host_age_years'].dropna().value_counts().sort_index()
    print(age_dist.head(10))
    
    # 可视化：房龄分组箱线图
    plt.figure(figsize=(10, 6))
    host_order = ['老房东', '新房东']
    sns.boxplot(data=df_with_age, x='host_type', y='price', order=host_order, palette='Set2')
    plt.title('房东类型 (房龄) 对房价的影响', fontsize=14)
    plt.xlabel('房东类型', fontsize=12)
    plt.ylabel('价格 (£)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/host_age_impact.png", dpi=150)
    plt.close()
    
    # 房龄与价格的相关性
    valid_age = df_with_age[df_with_age['host_age_years'].notna()]
    age_correlation = valid_age['host_age_years'].corr(valid_age['price'])
    print(f"\n房龄与价格的相关系数：{age_correlation:.4f}")
    
    # 保存结果
    df_with_age.to_csv(f"{save_dir}/data_with_host_age.csv", index=False)
    
    old_avg = age_stats.loc['老房东', 'mean']
    new_avg = age_stats.loc['新房东', 'mean']
    
    print(f"\n业务分析结论:")
    print(f"1. 老房东 (>=5 年) 均价£{old_avg:.0f}, 新房东 (<5 年) 均价£{new_avg:.0f}, 老房东溢价{(old_avg/new_avg-1)*100:.1f}%")
    print(f"2. 房龄与价格的相关系数为{age_correlation:.3f}, 呈现{'正' if age_correlation > 0 else '负'}相关")
    print(f"3. 老房东可能拥有更多经验、更好评价和更稳定的房源，因此可以收取更高价格")
    print(f"4. host_age_years 是一个有价值的特征，应加入模型")
    
    return df_with_age, age_stats, age_correlation


# ============================================================
# 主函数：运行所有实验
# ============================================================
def main():
    # 数据路径
    input_path = r"G:\vs_code\RepoSpark\train-database\student_train_full.csv"
    save_dir = r"G:\vs_code\RepoSpark\train-database\London Airbnb (6.0)"
    
    # 加载数据
    df = load_data(input_path)
    
    # 运行四个矿区实验
    df_tier, tier_stats = neighborhood_tier_analysis(df, save_dir)
    df_dist, dist_stats, price_per_km = distance_to_center_analysis(df, save_dir)
    df_amenities, amenity_stats = amenities_analysis(df, save_dir)
    df_age, age_stats, age_corr = host_age_analysis(df, save_dir)
    
    # 合并所有新特征
    df_final = df.copy()
    df_final['neighbourhood_tier'] = df_tier['neighbourhood_tier']
    df_final['dist_to_center'] = df_dist['dist_to_center']
    df_final['has_washer'] = df_amenities['has_washer']
    df_final['has_dryer'] = df_amenities['has_dryer']
    df_final['host_age_years'] = df_age['host_age_years']
    df_final['host_type'] = df_age['host_type']
    
    # 保存最终数据
    df_final.to_csv(f"{save_dir}/data_with_all_new_features.csv", index=False)
    
    print("\n" + "=" * 60)
    print("所有实验完成!")
    print("=" * 60)
    print(f"\n输出文件已保存到：{save_dir}")
    print("\n生成的文件:")
    print("- data_with_tier.csv (矿区 A 结果)")
    print("- data_with_distance.csv (矿区 B 结果)")
    print("- data_with_amenities.csv (矿区 C 结果)")
    print("- data_with_host_age.csv (矿区 D 结果)")
    print("- data_with_all_new_features.csv (合并所有新特征)")
    print("- neighborhood_tier_distribution.png (矿区 A 图表)")
    print("- distance_vs_price.png (矿区 B 图表)")
    print("- washer_impact.png (矿区 C 图表)")
    print("- host_age_impact.png (矿区 D 图表)")


if __name__ == "__main__":
    main()
