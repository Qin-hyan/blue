"""
类别型特征编码并合并到完整数据集
将编码后的类别特征与原始数值特征合并，生成完整的数值型训练矩阵
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def get_numeric_columns(df):
    """获取所有数值型列"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols


def get_categorical_columns(df, max_unique=100, exclude_cols=None):
    """获取类别型列"""
    if exclude_cols is None:
        exclude_cols = []
    
    categorical_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            if unique_count < max_unique:
                categorical_cols.append(col)
    return categorical_cols


def encode_categorical_features(df, rare_threshold=10, target_column='price'):
    """
    对类别型特征进行编码
    
    Args:
        df: 输入数据框
        rare_threshold: 稀有类别阈值
        target_column: 目标列名
        
    Returns:
        encoded_df: 编码后的特征数据框
        encoding_summary: 编码摘要信息
    """
    encoding_summary = {}
    encoded_features = []
    
    # 获取类别型列
    exclude_cols = [target_column] if target_column in df.columns else []
    categorical_cols = get_categorical_columns(df, exclude_cols=exclude_cols)
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        col_encoded = None
        encoding_method = None
        
        try:
            if unique_count <= 4:
                # 低基数：独热编码
                dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
                encoded_features.append(dummies)
                encoding_method = f"one_hot ({len(dummies.columns)} columns)"
                encoding_summary[col] = {
                    'method': 'one_hot',
                    'output_columns': dummies.columns.tolist(),
                    'unique_values': unique_count
                }
                
            elif unique_count <= 50 and target_column and target_column in df.columns:
                # 中基数且有目标列：目标编码
                series = df[col]
                target = df[target_column]
                
                stats = target.groupby(series).agg(['mean', 'count'])
                global_mean = target.mean()
                smoothing = 10
                stats['smoothed_mean'] = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
                
                encoded = series.map(stats['smoothed_mean'].to_dict()).fillna(global_mean)
                encoded_features.append(encoded.to_frame(name=f'{col}_target_encoded'))
                encoding_method = f"target_encoded"
                encoding_summary[col] = {
                    'method': 'target_encoding',
                    'output_columns': [f'{col}_target_encoded'],
                    'unique_values': unique_count,
                    'global_mean': global_mean
                }
                
            elif unique_count <= 50:
                # 中基数无目标列：频率编码
                freq_map = df[col].value_counts(normalize=True).to_dict()
                encoded = df[col].map(freq_map).fillna(0)
                encoded_features.append(encoded.to_frame(name=f'{col}_freq_encoded'))
                encoding_method = f"frequency_encoded"
                encoding_summary[col] = {
                    'method': 'frequency_encoding',
                    'output_columns': [f'{col}_freq_encoded'],
                    'unique_values': unique_count
                }
                
            else:
                # 高基数：先合并稀有类别再目标编码
                value_counts = df[col].value_counts()
                rare_categories = value_counts[value_counts < rare_threshold].index.tolist()
                
                series = df[col].copy()
                series = series.apply(lambda x: 'Other' if x in rare_categories else x)
                
                if target_column and target_column in df.columns:
                    target = df[target_column]
                    stats = target.groupby(series).agg(['mean', 'count'])
                    global_mean = target.mean()
                    smoothing = 10
                    stats['smoothed_mean'] = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
                    
                    encoded = series.map(stats['smoothed_mean'].to_dict()).fillna(global_mean)
                    encoded_features.append(encoded.to_frame(name=f'{col}_target_encoded'))
                    encoding_method = f"target_encoded (with rare category merging)"
                    encoding_summary[col] = {
                        'method': 'target_encoding_with_rare_merge',
                        'output_columns': [f'{col}_target_encoded'],
                        'unique_values': unique_count,
                        'rare_categories_merged': len(rare_categories),
                        'global_mean': global_mean
                    }
                else:
                    freq_map = series.value_counts(normalize=True).to_dict()
                    encoded = series.map(freq_map).fillna(0)
                    encoded_features.append(encoded.to_frame(name=f'{col}_freq_encoded'))
                    encoding_method = f"frequency_encoded (with rare category merging)"
                    encoding_summary[col] = {
                        'method': 'frequency_encoding_with_rare_merge',
                        'output_columns': [f'{col}_freq_encoded'],
                        'unique_values': unique_count,
                        'rare_categories_merged': len(rare_categories)
                    }
                    
        except Exception as e:
            print(f"  Warning: Encoding failed for {col}: {e}")
            encoding_summary[col] = {
                'method': 'failed',
                'error': str(e)
            }
    
    if encoded_features:
        encoded_df = pd.concat(encoded_features, axis=1)
    else:
        encoded_df = pd.DataFrame()
    
    return encoded_df, encoding_summary


def main():
    """主函数"""
    print("=" * 60)
    print("Categorical Feature Encoding and Merge")
    print("=" * 60)
    
    # 文件路径
    input_file = r"G:\vs_code\RepoSpark\train-database\London Airbnb (5.0)\cleaned_train_data.csv"
    output_file = r"G:\vs_code\RepoSpark\train-database\London Airbnb (5.0)\train_matrix_encoded.csv"
    
    # 读取数据
    print(f"\nReading data: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8')
    print(f"Data shape: {df.shape}")
    
    # 获取数值型列
    numeric_cols = get_numeric_columns(df)
    print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols[:10]}...")
    
    # 提取数值特征
    numeric_features = df[numeric_cols].copy()
    
    # 编码类别型特征
    print("\nEncoding categorical features...")
    encoded_features, encoding_summary = encode_categorical_features(df, rare_threshold=10)
    
    print(f"\nEncoded features shape: {encoded_features.shape}")
    print(f"Encoding summary:")
    for col, info in encoding_summary.items():
        print(f"  {col}: {info['method']}")
    
    # 合并数值特征和编码后的类别特征
    if not encoded_features.empty:
        # 确保行顺序一致
        encoded_features.index = numeric_features.index
        
        # 合并
        final_matrix = pd.concat([numeric_features, encoded_features], axis=1)
    else:
        final_matrix = numeric_features
    
    print(f"\nFinal matrix shape: {final_matrix.shape}")
    print(f"Total features: {final_matrix.shape[1]}")
    print(f"Total samples: {final_matrix.shape[0]}")
    
    # 保存结果
    final_matrix.to_csv(output_file, index=False)
    print(f"\nFinal matrix saved: {output_file}")
    
    # 打印前几行
    print(f"\nFirst 5 rows:")
    print(final_matrix.head())
    
    # 打印列信息
    print(f"\nAll columns ({final_matrix.shape[1]}):")
    for i, col in enumerate(final_matrix.columns):
        print(f"  {i+1}. {col}")


if __name__ == "__main__":
    main()