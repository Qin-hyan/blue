import pandas as pd
import numpy as np

df = pd.read_csv('student_train_full.csv', nrows=100)

print("=== host_response_rate (前30行) ===")
for i in range(min(30, len(df))):
    print(f"  [{i}] = {repr(df['host_response_rate'].iloc[i])}")

print("\n=== host_since (前20行) ===")
for i in range(min(20, len(df))):
    print(f"  [{i}] = {repr(df['host_since'].iloc[i])}")

print("\n=== price 统计 ===")
print(df['price'].describe())
print(f"唯一值数量: {df['price'].nunique()}")
print(f"非空数量: {df['price'].notna().sum()}")

print("\n=== bedrooms/beds/bathrooms 缺失值 ===")
print(f"bedrooms: 缺失 {df['bedrooms'].isna().sum()}, 非空 {df['bedrooms'].notna().sum()}")
print(f"beds: 缺失 {df['beds'].isna().sum()}, 非空 {df['beds'].notna().sum()}")
print(f"bathrooms: 缺失 {df['bathrooms'].isna().sum()}, 非空 {df['bathrooms'].notna().sum()}")

print("\n=== bedrooms/beds/bathrooms 样本 ===")
for i in range(min(20, len(df))):
    print(f"  [{i}] bedrooms={repr(df['bedrooms'].iloc[i])}, beds={repr(df['beds'].iloc[i])}, bathrooms={repr(df['bathrooms'].iloc[i])}")

print("\n=== price 样本 (前30行) ===")
for i in range(min(30, len(df))):
    print(f"  [{i}] = {repr(df['price'].iloc[i])}")