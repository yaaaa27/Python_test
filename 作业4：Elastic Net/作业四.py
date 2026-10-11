import os
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---------- 0. 路径 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'netflix_titles.csv')

# ---------- 1. 读数据 ----------
df = pd.read_csv(CSV_PATH)
print("原始数据形状：", df.shape)
print("列名：", list(df.columns))

# ---------- 2. 只保留电影，提取时长 ----------
df = df[df['type'] == 'Movie'].copy()
df['minutes'] = df['duration'].str.extract(r'(\d+)').astype(float)
df = df.dropna(subset=['minutes'])
print("电影样本数：", len(df))

# ---------- 3. 多值列 One-Hot ----------
def explode_dummies(series, prefix, top_n=None):
    s = series.dropna().str.split(', ')
    exploded = s.explode()
    if top_n:
        keep = exploded.value_counts().head(top_n).index
        exploded = exploded[exploded.isin(keep)]
    dummies = pd.get_dummies(exploded, prefix=prefix)
    return dummies.groupby(level=0).sum()

genre_d   = explode_dummies(df['listed_in'], 'genre')
country_d = explode_dummies(df['country'],   'country', top_n=20)
rating_d  = pd.get_dummies(df['rating'],     prefix='rating')

# ---------- 4. 数值特征 ----------
num = pd.DataFrame(index=df.index)
num['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
num['year_added']   = pd.to_datetime(df['date_added'], errors='coerce').dt.year

# 分别用中位数填补（年份合理）
num['release_year'] = num['release_year'].fillna(num['release_year'].median())
num['year_added']   = num['year_added'].fillna(num['year_added'].median())

# ---------- 5. 合并特征矩阵 ----------
X = pd.concat([
    num,
    genre_d.reindex(df.index).fillna(0),
    country_d.reindex(df.index).fillna(0),
    rating_d.reindex(df.index).fillna(0)
], axis=1)

X = X.loc[:, X.sum() > 5]

nan_count = X.isna().sum().sum()
print("合并后 X 中 NaN 总数：", nan_count)
if nan_count > 0:
    print("含 NaN 的列：")
    print(X.isna().sum()[X.isna().sum() > 0])
    X = X.fillna(0)   # 数值列填 0，One-Hot 列本就该是 0

y = df['minutes']

print("特征矩阵形状：", X.shape)

# ---------- 7. 划分 + 标准化 ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 再确认一次
assert not np.isnan(X_train_s).any(), "训练集仍有 NaN"
assert not np.isnan(X_test_s).any(),  "测试集仍有 NaN"

# ---------- 8. Elastic Net ----------
model = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
    cv=5,
    max_iter=10000,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_s, y_train)

# ---------- 9. 评估 ----------
pred = model.predict(X_test_s)
print("\n========== 模型结果 ==========")
print(f"最优 alpha     = {model.alpha_:.4f}")
print(f"最优 l1_ratio  = {model.l1_ratio_}")
print(f"测试集 R²      = {r2_score(y_test, pred):.3f}")
print(f"测试集 RMSE    = {np.sqrt(mean_squared_error(y_test, pred)):.1f} 分钟")

# ---------- 10. 查看被选中的特征 ----------
coef = pd.Series(model.coef_, index=X.columns)
selected = coef[coef != 0].sort_values(key=abs, ascending=False)

print(f"\n共 {len(X.columns)} 个特征，被 Elastic Net 选中 {len(selected)} 个")
print("\n========== 系数绝对值 Top 20 ==========")
print(selected.head(20).to_string())

pos = selected[selected > 0].head(10)
neg = selected[selected < 0].head(10)
print("\n--- 使电影更长（正系数）Top 10 ---")
print(pos.to_string())
print("\n--- 使电影更短（负系数）Top 10 ---")
print(neg.to_string())
printf("\n--- 分析结束 ---")
