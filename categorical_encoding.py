"""
Categorical Encoding Script
For encoding categorical features in London Airbnb dataset

Encoding methods:
1. One-Hot Encoding - for low-cardinality nominal categories
2. Label Encoding - for binary/ordinal categories
3. Target Encoding - for high-cardinality categories
4. Frequency Encoding - alternative to target encoding
5. Rare Category Handling - merge low-frequency categories to "Other"
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class CategoricalEncoder:
    """类别型特征编码器"""
    
    def __init__(self, rare_threshold=10, target_column='price'):
        """
        初始化编码器
        
        Args:
            rare_threshold: 稀有类别阈值，出现次数低于此值的类别将被归为"Other"
            target_column: 目标列名（用于目标编码）
        """
        self.rare_threshold = rare_threshold
        self.target_column = target_column
        self.encoders = {}  # 存储每个特征的编码映射
        self.target_encoders = {}  # 存储目标编码的映射
        
    def _handle_rare_categories(self, series, column_name):
        """
        处理稀有类别：将低频类别归为"Other"
        
        Args:
            series: 要处理的 Series
            column_name: 列名
            
        Returns:
            处理后的 Series 和编码映射
        """
        value_counts = series.value_counts()
        rare_categories = value_counts[value_counts < self.rare_threshold].index.tolist()
        
        # 创建映射
        mapping = {}
        for cat in series.unique():
            if cat in rare_categories or pd.isna(cat):
                mapping[cat] = 'Other'
            else:
                mapping[cat] = cat
        
        encoded = series.map(mapping).fillna('Other')
        self.encoders[column_name] = mapping
        self.encoders[f'{column_name}_rare_count'] = len(rare_categories)
        
        return encoded, rare_categories
    
    def one_hot_encode(self, df, column_name, handle_rare=True):
        """
        独热编码
        
        Args:
            df: 数据框
            column_name: 要编码的列名
            handle_rare: 是否先处理稀有类别
            
        Returns:
            编码后的数据框
        """
        series = df[column_name].astype(str)
        
        if handle_rare:
            series, rare_cats = self._handle_rare_categories(series, column_name)
        
        # 创建独热编码
        dummies = pd.get_dummies(series, prefix=column_name, dummy_na=False)
        
        # 存储编码信息
        self.encoders[f'{column_name}_categories'] = dummies.columns.tolist()
        
        return dummies
    
    def label_encode(self, df, column_name):
        """
        标签编码：将类别转换为 0, 1, 2, ...
        
        Args:
            df: 数据框
            column_name: 要编码的列名
            
        Returns:
            编码后的 Series
        """
        series = df[column_name]
        
        # 创建排序后的类别列表
        unique_cats = series.dropna().unique()
        cat_to_int = {cat: idx for idx, cat in enumerate(unique_cats)}
        
        # 存储编码映射
        self.encoders[column_name] = cat_to_int
        self.encoders[f'{column_name}_reverse'] = {idx: cat for cat, idx in cat_to_int.items()}
        
        # 编码
        encoded = series.map(cat_to_int).fillna(-1)  # NaN 映射为 -1
        
        return encoded
    
    def binary_encode(self, df, column_name, true_value='t'):
        """
        二元编码：将二元类别转换为 0/1
        
        Args:
            df: 数据框
            column_name: 要编码的列名
            true_value: 表示"真"的值（默认't'）
            
        Returns:
            编码后的 Series
        """
        self.encoders[column_name] = {true_value: 1, 'f': 0, 'False': 0, 'True': 1, 't': 1, 'f': 0}
        self.encoders[f'{column_name}_true_value'] = true_value
        
        encoded = df[column_name].map({true_value: 1, 'f': 0, 'False': 0, 'True': 1, 't': 1})
        return encoded.fillna(0)
    
    def target_encode(self, df, column_name, smoothing=10):
        """
        目标编码：用该类别对应的目标变量均值替换
        
        Args:
            df: 数据框
            column_name: 要编码的列名
            smoothing: 平滑参数，越大越倾向于全局均值
            
        Returns:
            编码后的 Series
        """
        if self.target_column not in df.columns:
            raise ValueError(f"目标列 '{self.target_column}' 不在数据框中")
        
        series = df[column_name]
        target = df[self.target_column]
        
        # 计算每个类别的目标均值和计数
        stats = target.groupby(series).agg(['mean', 'count'])
        
        # 应用平滑
        global_mean = target.mean()
        stats['smoothed_mean'] = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        
        # 存储编码映射
        self.target_encoders[column_name] = stats['smoothed_mean'].to_dict()
        self.target_encoders[f'{column_name}_global_mean'] = global_mean
        
        # 编码
        encoded = series.map(self.target_encoders[column_name]).fillna(global_mean)
        
        return encoded
    
    def frequency_encode(self, df, column_name):
        """
        频率编码：用该类别的出现频率替换
        
        Args:
            df: 数据框
            column_name: 要编码的列名
            
        Returns:
            编码后的 Series
        """
        series = df[column_name]
        freq_map = series.value_counts(normalize=True).to_dict()
        
        # 存储编码映射
        self.encoders[f'{column_name}_freq_map'] = freq_map
        
        # 编码
        encoded = series.map(freq_map).fillna(0)
        
        return encoded
    
    def extract_property_type_features(self, df, column_name='property_type'):
        """
        从 property_type 提取结构化特征
        property_type 格式如 "Entire home/apt", "Private room in house"
        
        Args:
            df: 数据框
            column_name: 列名
            
        Returns:
            提取的特征数据框
        """
        series = df[column_name].astype(str)
        
        # 提取房间类型前缀 (Entire/Private/Shared/Room)
        room_prefix = series.str.extract(r'^(Entire|Private|Shared|Room)', expand=False)
        room_prefix = room_prefix.fillna('Unknown')
        
        # 提取房产类型后缀
        property_suffix = series.str.replace(r'^(Entire|Private|Shared|Room)\s*', '', regex=True)
        property_suffix = property_suffix.str.replace(r'\s+in\s+.*$', '', regex=True)
        
        # 统计各类别频率
        prefix_counts = room_prefix.value_counts()
        suffix_counts = property_suffix.value_counts()
        
        # 合并低频前缀
        rare_prefix = prefix_counts[prefix_counts < self.rare_threshold].index.tolist()
        room_prefix = room_prefix.apply(lambda x: 'Other' if x in rare_prefix else x)
        
        # 合并低频后缀
        rare_suffix = suffix_counts[suffix_counts < self.rare_threshold].index.tolist()
        property_suffix = property_suffix.apply(lambda x: 'Other' if x in rare_suffix else x)
        
        # 独热编码前缀
        prefix_dummies = pd.get_dummies(room_prefix, prefix='room_type')
        
        # 对后缀进行目标编码或频率编码
        suffix_freq = self.frequency_encode(pd.DataFrame({column_name: property_suffix}), column_name)
        
        features = pd.concat([prefix_dummies, suffix_freq.rename('property_suffix_freq')], axis=1)
        
        self.encoders[f'{column_name}_prefix_categories'] = prefix_dummies.columns.tolist()
        
        return features


def encode_dataset(input_path, output_path, encoder_path=None):
    """
    Encode categorical features in the dataset
    
    Args:
        input_path: Input file path
        output_path: Output file path
        encoder_path: Encoder save path (optional)
    """
    print("=" * 60)
    print("Categorical Feature Encoding")
    print("=" * 60)
    
    # Read data
    print(f"\nReading data: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8')
    print(f"Data shape: {df.shape}")
    
    # Define target column (if exists)
    target_column = 'price' if 'price' in df.columns else None
    
    # Initialize encoder
    encoder = CategoricalEncoder(rare_threshold=10, target_column=target_column)
    
    # Define encoding config for categorical features
    encoding_config = {
        'room_type': ('one_hot', True),
        'host_is_superhost': ('binary', False),
        'host_has_profile_pic': ('binary', False),
        'host_identity_verified': ('binary', False),
        'has_availability': ('binary', False),
        'instant_bookable': ('binary', False),
        'neighbourhood': ('target', True),
        'neighbourhood_cleansed': ('target', True),
        'property_type': ('custom', True),
    }
    
    # Identify categorical columns in data
    categorical_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            if unique_count < 100:
                categorical_columns.append(col)
    
    print(f"\nIdentified categorical columns: {categorical_columns}")
    
    # Store encoded features
    encoded_features = []
    processed_columns = []
    
    # Process each categorical column
    for col in categorical_columns:
        print(f"\nProcessing column: {col} (unique values: {df[col].nunique()})")
        
        if col not in encoding_config:
            unique_count = df[col].nunique()
            if unique_count <= 5:
                config = ('one_hot', True) if unique_count <= 3 else ('label', True)
            elif unique_count <= 20:
                config = ('one_hot', True)
            else:
                config = ('target' if target_column else 'frequency', True)
        else:
            config = encoding_config[col]
        
        method, handle_rare = config
        
        try:
            if method == 'one_hot':
                dummies = encoder.one_hot_encode(df, col, handle_rare)
                encoded_features.append(dummies)
                processed_columns.extend(dummies.columns.tolist())
                print(f"  -> One-Hot Encoding, generated {len(dummies.columns)} columns")
                
            elif method == 'label':
                encoded = encoder.label_encode(df, col)
                encoded_features.append(encoded.to_frame())
                processed_columns.append(col)
                print(f"  -> Label Encoding")
                
            elif method == 'binary':
                encoded = encoder.binary_encode(df, col)
                encoded_features.append(encoded.to_frame())
                processed_columns.append(col)
                print(f"  -> Binary Encoding")
                
            elif method == 'target' and target_column:
                encoded = encoder.target_encode(df, col)
                encoded_features.append(encoded.to_frame())
                processed_columns.append(f'{col}_target_encoded')
                print(f"  -> Target Encoding (smoothed)")
                
            elif method == 'frequency':
                encoded = encoder.frequency_encode(df, col)
                encoded_features.append(encoded.to_frame())
                processed_columns.append(f'{col}_freq_encoded')
                print(f"  -> Frequency Encoding")
                
            elif method == 'custom':
                if col == 'property_type':
                    custom_features = encoder.extract_property_type_features(df, col)
                    encoded_features.append(custom_features)
                    processed_columns.extend(custom_features.columns.tolist())
                    print(f"  -> Custom Encoding, generated {len(custom_features.columns)} columns")
                    
        except Exception as e:
            print(f"  -> Encoding failed: {e}")
    
    # Merge all encoded features
    if encoded_features:
        encoded_df = pd.concat(encoded_features, axis=1)
        print(f"\nEncoded features shape: {encoded_df.shape}")
        print(f"Generated columns: {processed_columns}")
        
        # Save encoded data
        encoded_df.to_csv(output_path, index=False)
        print(f"\nEncoded results saved: {output_path}")
        
        # Save encoder (for test set transformation)
        if encoder_path:
            import pickle
            with open(encoder_path, 'wb') as f:
                pickle.dump(encoder, f)
            print(f"Encoder saved: {encoder_path}")
        
        return encoded_df
    else:
        print("\nNo encoded features generated")
        return None


def main():
    """Main function"""
    # File paths
    input_file = r"G:\vs_code\RepoSpark\train-database\London Airbnb (5.0)\cleaned_train_data.csv"
    output_file = r"G:\vs_code\RepoSpark\train-database\London Airbnb (5.0)\encoded_categorical_features.csv"
    encoder_file = r"G:\vs_code\RepoSpark\train-database\London Airbnb (5.0)\encoder.pkl"
    
    # Execute encoding
    encoded_df = encode_dataset(input_file, output_file, encoder_file)
    
    # Print statistics
    if encoded_df is not None:
        print("\n" + "=" * 60)
        print("Encoding Statistics")
        print("=" * 60)
        print(f"Total features: {encoded_df.shape[1]}")
        print(f"Total samples: {encoded_df.shape[0]}")
        print(f"\nFirst 10 columns:\n{encoded_df.columns[:10].tolist()}")
        print(f"\nFirst 5 rows:\n{encoded_df.head()}")


if __name__ == "__main__":
    main()