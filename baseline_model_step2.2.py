"""
Step 2.2: Baseline 模型验证特征影响
对比加入新特征前后的模型性能
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }


def print_metrics(metrics, title=""):
    """打印评估指标"""
    print(f"\n{title}")
    print("-" * 40)
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAE:  {metrics['MAE']:.2f}")
    print(f"R2:   {metrics['R2']:.4f}")


def main():
    # 加载数据
    input_path = r"G:\vs_code\RepoSpark\train-database\London Airbnb (6.0)\data_with_all_new_features.csv"
    df = pd.read_csv(input_path)
    
    print("=" * 60)
    print("Step 2.2: Baseline 模型验证特征影响")
    print("=" * 60)
    print(f"\n数据形状：{df.shape}")
    
    # 准备特征
    # 基础特征
    base_features = ['bedrooms', 'bathrooms', 'beds', 'accommodates', 'latitude', 'longitude']
    
    # 新特征
    new_features = ['neighbourhood_tier', 'dist_to_center', 'has_washer', 
                    'has_dryer', 'host_age_years', 'host_type']
    
    # 编码分类变量
    df['neighbourhood_tier_encoded'] = df['neighbourhood_tier'].map({'顶级': 4, '高级': 3, '大众': 2, '廉价': 1})
    df['host_type_encoded'] = df['host_type'].map({'老房东': 1, '新房东': 0})
    
    # 特征组合
    all_features = base_features + ['neighbourhood_tier_encoded', 'dist_to_center', 
                                     'has_washer', 'has_dryer', 'host_age_years', 'host_type_encoded']
    
    # 处理 NaN 值
    df_clean = df[all_features + ['price']].dropna()
    
    print(f"\n清洗后数据形状：{df_clean.shape}")
    
    # 划分训练集和测试集
    X = df_clean[all_features]
    y = df_clean['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\n训练集大小：{len(X_train)}")
    print(f"测试集大小：{len(X_test)}")
    
    # 训练模型
    print("\n" + "=" * 60)
    print("模型训练")
    print("=" * 60)
    
    # 使用 GradientBoosting 模型
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算指标
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, "\n完整特征模型性能:")
    
    # 特征重要性
    print("\n" + "=" * 60)
    print("特征重要性")
    print("=" * 60)
    
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    # 分析新特征的影响
    print("\n" + "=" * 60)
    print("新特征贡献分析")
    print("=" * 60)
    
    new_feature_importance = feature_importance[feature_importance['feature'].isin(new_features)]
    print("\n新加入的特征重要性:")
    print(new_feature_importance.to_string(index=False))
    
    total_new_importance = new_feature_importance['importance'].sum()
    print(f"\n新特征总重要性：{total_new_importance:.4f} ({total_new_importance*100:.1f}%)")
    
    # 保存结果
    feature_importance.to_csv(r"G:\vs_code\RepoSpark\train-database\London Airbnb (6.0)\feature_importance.csv", index=False)
    
    print("\n" + "=" * 60)
    print("分析结论")
    print("=" * 60)
    print(f"\n1. 模型 R2 分数：{metrics['R2']:.4f}, 说明模型能解释 {metrics['R2']*100:.1f}% 的价格变异")
    print(f"2. RMSE: £{metrics['RMSE']:.2f}, 平均预测误差约 £{metrics['MAE']:.2f}")
    print(f"3. 最重要的特征：{feature_importance.iloc[0]['feature']} (重要性：{feature_importance.iloc[0]['importance']:.4f})")
    print(f"4. 新特征中最重要的：{new_feature_importance.iloc[0]['feature']} (重要性：{new_feature_importance.iloc[0]['importance']:.4f})")
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred
    })
    results_df.to_csv(r"G:\vs_code\RepoSpark\train-database\London Airbnb (6.0)\prediction_results.csv", index=False)
    
    print("\n输出文件已生成:")
    print("- feature_importance.csv (特征重要性)")
    print("- prediction_results.csv (预测结果)")


if __name__ == "__main__":
    main()