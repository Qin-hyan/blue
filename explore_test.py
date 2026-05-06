import pandas as pd

df = pd.read_csv('public_test_questions.csv', nrows=100)

print('=== 测试集列名 ===')
for i, c in enumerate(df.columns):
    print(f'  {i}: {c}')

print(f'\n=== 测试集形状: {df.shape} ===')
print(f'bedrooms 非空: {df["bedrooms"].notna().sum()}/{len(df)}')
print(f'beds 非空: {df["beds"].notna().sum()}/{len(df)}')
print(f'bathrooms 非空: {df["bathrooms"].notna().sum()}/{len(df)}')
print(f'host_response_rate 非空: {df["host_response_rate"].notna().sum()}/{len(df)}')
print(f'host_since 非空: {df["host_since"].notna().sum()}/{len(df)}')

# 检查测试集是否有 price
if 'price' in df.columns:
    print(f'\nprice 非空: {df["price"].notna().sum()}/{len(df)}')
else:
    print('\n测试集没有 price 列！')

# host_response_rate 样本
print(f'\nhost_response_rate 样本:')
for i in range(min(10, len(df))):
    print(f'  [{i}] = {repr(df["host_response_rate"].iloc[i])}')
