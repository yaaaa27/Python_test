import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, lasso_path, ridge_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 0. 通用函数：1-SE 法则
# ============================================================
def one_se_rule(cv_model, alphas, mse_mean, mse_std):
    """
    返回 (lambda_min, lambda_1se)
    lambda_min: CV-MSE 最小对应的 alpha
    lambda_1se: 在 MSE_min + SE 范围内最大的 alpha（最稀疏模型）
    """
    idx_min = np.argmin(mse_mean)
    alpha_min = alphas[idx_min]
    threshold = mse_mean[idx_min] + mse_std[idx_min]
    # 在阈值范围内的最大 alpha
    valid = alphas[mse_mean <= threshold]
    alpha_1se = valid.max() if len(valid) > 0 else alpha_min
    return alpha_min, alpha_1se


# ============================================================
# 1. 数据加载
# ============================================================
def load_hitters(path='Hitters.csv'):
    df = pd.read_csv(path).dropna()
    y = df['Salary'].values
    X = df.drop(columns=['Salary'])
    X = pd.get_dummies(X, drop_first=True)
    return X, y

def load_boston(path='housing.csv'):
    cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',
            'RAD','TAX','PTRATIO','B','LSTAT','MEDV']
    df = pd.read_csv(path, sep=r'\s+', header=None, names=cols)
    y = df['MEDV'].values
    X = df.drop(columns=['MEDV'])
    return X, y


# ============================================================
# 2. 单个数据集的分析流程
# ============================================================
def analyze(X, y, name, alphas=np.logspace(-3, 3, 100), l1_ratios=(0.1, 0.5, 0.9, 1.0)):
    print(f"\n{'='*70}\n数据集：{name}   n={len(y)}, p={X.shape[1]}\n{'='*70}")
    feature_names = list(X.columns)

    # 划分训练/测试
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y, test_size=0.3, random_state=42)

    # 标准化
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ---------- 2.1 RidgeCV ----------
    ridge = RidgeCV(alphas=alphas, cv=10, scoring='neg_mean_squared_error')
    ridge.fit(X_tr_s, y_tr)
    rmse_ridge = np.sqrt(mean_squared_error(y_te, ridge.predict(X_te_s)))
    nz_ridge = np.sum(np.abs(ridge.coef_) > 1e-8)

    # ---------- 2.2 LassoCV ----------
    lasso = LassoCV(alphas=alphas, cv=10, max_iter=100000, random_state=42)
    lasso.fit(X_tr_s, y_tr)
    rmse_lasso = np.sqrt(mean_squared_error(y_te, lasso.predict(X_te_s)))
    nz_lasso = np.sum(np.abs(lasso.coef_) > 1e-8)

    # ---------- 2.3 ElasticNetCV ----------
    enet = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=10,
                        max_iter=100000, random_state=42)
    enet.fit(X_tr_s, y_tr)
    rmse_enet = np.sqrt(mean_squared_error(y_te, enet.predict(X_te_s)))
    nz_enet = np.sum(np.abs(enet.coef_) > 1e-8)

    # ---------- 2.4 汇总 ----------
    print(f"{'模型':<15}{'最优α/λ':<15}{'测试RMSE':<12}{'非零变量数':<10}")
    print(f"{'Ridge':<15}{ridge.alpha_:<15.4f}{rmse_ridge:<12.4f}{nz_ridge:<10}")
    print(f"{'Lasso':<15}{lasso.alpha_:<15.4f}{rmse_lasso:<12.4f}{nz_lasso:<10}")
    print(f"{'ElasticNet':<15}{enet.alpha_:<15.4f}{rmse_enet:<12.4f}{nz_enet:<10}"
          f"  (l1_ratio={enet.l1_ratio_})")

    # ---------- 2.5 系数路径图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Lasso 路径
    alphas_path, coefs_lasso, _ = lasso_path(X_tr_s, y_tr, alphas=alphas)
    for i in range(coefs_lasso.shape[0]):
        axes[0].plot(np.log10(alphas_path), coefs_lasso[i], lw=1.2)
    axes[0].axvline(np.log10(lasso.alpha_), color='r', ls='--',
                    label=f'λ.min={lasso.alpha_:.3f}')
    axes[0].set_title(f'{name} · Lasso 系数路径')
    axes[0].set_xlabel('log10(α)'); axes[0].set_ylabel('系数')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Ridge 路径
    ridge_alphas = np.logspace(-3, 3, 100)
    ridge_coefs = []
    for a in ridge_alphas:
        r = ridge_regression(X_tr_s, y_tr, alpha=a)
        ridge_coefs.append(r)
    ridge_coefs = np.array(ridge_coefs)
    for i in range(ridge_coefs.shape[1]):
        axes[1].plot(np.log10(ridge_alphas), ridge_coefs[:, i], lw=1.2)
    axes[1].axvline(np.log10(ridge.alpha_), color='r', ls='--',
                    label=f'λ={ridge.alpha_:.3f}')
    axes[1].set_title(f'{name} · Ridge 系数路径')
    axes[1].set_xlabel('log10(λ)'); axes[1].set_ylabel('系数')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'path_{name}.png', dpi=120)
    plt.show()

    # ---------- 2.6 Lasso CV 曲线 + 1-SE ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    mse_mean = lasso.mse_path_.mean(axis=1)
    mse_std = lasso.mse_path_.std(axis=1) / np.sqrt(10)
    ax.errorbar(np.log10(lasso.alphas_), mse_mean, yerr=mse_std,
                fmt='o-', ms=3, capsize=2, color='steelblue')
    a_min, a_1se = one_se_rule(lasso, lasso.alphas_, mse_mean, mse_std)
    ax.axvline(np.log10(a_min), color='r', ls='--', label=f'λ.min={a_min:.3f}')
    ax.axvline(np.log10(a_1se), color='g', ls='--', label=f'λ.1se={a_1se:.3f}')
    ax.set_title(f'{name} · Lasso 10折CV曲线')
    ax.set_xlabel('log10(α)'); ax.set_ylabel('CV-MSE')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f'cv_{name}.png', dpi=120); plt.show()

    # ---------- 2.7 1-SE 法则下的非零变量 ----------
    from sklearn.linear_model import Lasso
    lasso_1se = Lasso(alpha=a_1se, max_iter=100000).fit(X_tr_s, y_tr)
    rmse_1se = np.sqrt(mean_squared_error(y_te, lasso_1se.predict(X_te_s)))
    nz_1se = np.sum(np.abs(lasso_1se.coef_) > 1e-8)
    print(f"\n1-SE 法则：λ.1se = {a_1se:.4f}，测试RMSE = {rmse_1se:.4f}，"
          f"非零变量数 = {nz_1se}（vs λ.min 下 {nz_lasso} 个）")

    return {'Ridge': (ridge.alpha_, rmse_ridge, nz_ridge),
            'Lasso': (lasso.alpha_, rmse_lasso, nz_lasso),
            'ElasticNet': (enet.alpha_, rmse_enet, nz_enet),
            'Lasso_1SE': (a_1se, rmse_1se, nz_1se)}


# ============================================================
# 3. 运行两个数据集
# ============================================================
X_hit, y_hit = load_hitters('Hitters.csv')
res_hit = analyze(X_hit, y_hit, 'Hitters')

X_bos, y_bos = load_boston('housing.csv')
res_bos = analyze(X_bos, y_bos, 'Boston')
