# blue
AI science


类别型特征编码任务已完成！
## 生成的文件

1. **categorical_encoding.py** - 独立的类别型特征编码脚本
   - 支持多种编码方法：独热编码、标签编码、目标编码、频率编码、二元编码
   - 稀有类别处理功能
   - 编码器保存/加载功能

2. **encode_and_merge.py** - 编码并合并到完整数据集的脚本
   - 自动识别类别型特征
   - 根据基数自动选择编码方法
   - 合并数值特征和编码后的类别特征

3. **encoded_categorical_features.csv** - 仅包含编码后的类别特征 (31 列)

4. **train_matrix_encoded.csv** - 完整的训练矩阵 (71 列)
   - 45 个原始数值特征
   - 26 个编码后的类别特征
   - 42,619 个样本

## 编码方法总结

| 列名 | 编码方法 | 唯一值数 |
|------|----------|----------|
| source | one_hot | 1 |
| host_response_time | one_hot | 4 |
| host_is_superhost | one_hot | 2 |
| host_has_profile_pic | one_hot | 2 |
| host_identity_verified | one_hot | 2 |
| neighbourhood | one_hot | 1 |
| room_type | one_hot | 4 |
| has_availability | one_hot | 1 |
| last_scraped | target_encoding | 5 |
| host_response_rate | target_encoding_with_rare_merge | 90 |
| host_verifications | target_encoding | 8 |
| neighbourhood_cleansed | target_encoding | 33 |
| property_type | target_encoding_with_rare_merge | 75 |
| bathrooms_text | target_encoding | 40 |
| calendar_last_scraped | target_encoding | 5 |
| instant_bookable | one_hot | 2 |

## 编码结果
- 总特征数：71
- 总样本数：42,619
- 输出文件：`G:\vs_code\RepoSpark\train-database\London Airbnb (5.0)\train_matrix_encoded.csv`

London Airbnb 价格分析项目

项目内容：

四个实验分析：

矿区 A：地段的鄙视链（Neighborhood Tier 分箱）

矿区 B：黄金一公里（距离地铁站距离分析）

矿区 C：洗衣机自由（设施影响分析）

矿区 D：姜是老的辣（Host 房龄影响分析）

核心文件：

feature_engineering_step2.2.py - 特征工程脚本

baseline_model_step2.2.py - Baseline 模型验证

分析报告.md - 完整分析报告

可视化图表：distance_vs_price.png, host_age_impact.png, neighborhood_tier_distribution.png, washer_impact.png


注意事项：

大型数据文件（>100MB）已通过 .gitignore 排除，避免超过 GitHub 限制

原始数据文件需要本地运行特征工程脚本生成
