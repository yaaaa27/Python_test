import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. 加载数据
# 假设您的数据文件名为 Carseats.csv
file_path = 'Carseats.csv'
df = pd.read_csv(file_path)

print("数据加载成功，总行数:", len(df))

# 2. 处理分类变量 (ShelveLoc)
# 题目要求以 ShelveLoc 作为定性特征。为了在回归中使用它，我们需要进行 One-Hot 编码（哑变量化）。
# 在 statsmodels 中，默认会将第一个类别（按字母顺序）作为基准组，并自动从变量名中体现出来。
# ShelveLoc 有三个类别: Bad, Good, Medium。字母顺序最靠前的是 Bad，因此 Bad 将成为基准组。
X = df[['Price', 'Income', 'Advertising', 'ShelveLoc']]
X = pd.get_dummies(X, columns=['ShelveLoc'], drop_first=True, dtype=float)
# drop_first=True 会删掉第一个基准组 (Bad)，剩下 ShelveLoc_Good 和 ShelveLoc_Medium

# 添加截距项 (Intercept)
X = sm.add_constant(X)
y = df['Sales']

# 3. 建立多元线性回归模型
model = sm.OLS(y, X).fit()

# 4. 打印模型拟合报告
print("\n================ 模型拟合报告 ================")
print(model.summary())

# 5. 解答题目具体问题
print("\n================ 题目解答 ================")

# --- 问题 1: 提取 ShelveLoc 的基准组是什么？---
print("【问题1】ShelveLoc 的基准组是: Bad")
print("解释: 在变量名中只出现了 ShelveLoc_Good 和 ShelveLoc_Medium，没有 ShelveLoc_Bad，说明 Bad 被当作了基准组。")

# --- 问题 2: 解读 ShelveLoc[Good] 系数的实际商业含义 ---
good_coef = model.params['ShelveLoc_Good']
print(f"\n【问题2】ShelveLoc[Good] 的系数为: {good_coef:.4f}")
print("商业含义解读:")
print(f"在控制价格 (Price)、收入 (Income)、广告投入 (Advertising) 等其他变量不变的情况下，")
print(f"将货架位置从 'Bad' (基准组，最差位置) 提升到 'Good' (好位置)，")
print(f"该产品的销售额 (Sales) 平均会增加 {good_coef:.4f} 个单位（即约 {good_coef*1000:.0f} 件）。")
print("这充分说明，在汽车座椅的销售中，货架位置对销量有着非常重要且正向的促进作用。")

# --- 问题 3: 计算各变量的 VIF，评估多重共线性风险 ---
print("\n【问题3】计算各变量的 VIF (方差膨胀因子):")
# 计算 VIF
vif_data = pd.DataFrame()
vif_data["变量名称"] = X.columns
vif_data["VIF 值"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# 去掉截距项的 VIF（因为截距项本身不是我们关心的解释变量）
vif_data = vif_data[vif_data["变量名称"] != "const"]
print(vif_data.to_string(index=False))

print("\n多重共线性风险评估:")
max_vif = vif_data['VIF 值'].max()
if max_vif > 10:
    print(f"警告: 最大 VIF 为 {max_vif:.2f} > 10，存在严重的多重共线性风险！")
elif max_vif > 5:
    print(f"注意: 最大 VIF 为 {max_vif:.2f}，存在一定程度的多重共线性，需要关注。")
else:
    print(f"结论: 所有变量的 VIF 均远小于 5 (最大仅为 {max_vif:.2f})，不存在多重共线性风险。")
