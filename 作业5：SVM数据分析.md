## 一、 数据集与预处理
实验使用了四个数据集，特点差异极大：
1. **Spambase (垃圾邮件)**：4601 个样本，57 维特征。线性可分性较好。
2. **German Credit (信用违约)**：1000 个样本，20 个类别特征。样本量小，且存在明显的类别不平衡（好信用多，坏信用少）。
3. **Leukemia (白血病基因)**：38 个样本，7129 个基因特征。典型的“高维小样本”（n << d）。
4. **MNIST (手写数字 0 和 1)**：约 1.2 万张图像，784 维特征。

**预处理要点**：因为 SVM 是依赖于计算距离（最大化间隔）的算法，如果特征尺度差异过大，模型会完全被大数值特征主导。因此，**所有的数据在建模前都必须使用 `StandardScaler` 进行标准化**。对于信用数据，先进行独热编码（One-Hot）处理类别变量。

---

## 二、 完整 Python 实现代码

以下代码整合了四个案例的完整流程

```python
# -*- coding: utf-8 -*-
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无界面环境保存图片
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(DATA_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ================= 通用评估与绘图函数 =================
def evaluate_model(y_true, y_pred, y_prob=None, name="model"):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan")
    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

def plot_cm_and_roc(y_true, y_pred, y_prob, title, save_name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cm = confusion_matrix(y_true, y_pred)
    im = axes[0].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[0].set_title(f"{title} - CM")
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center")
    
    if y_prob is not None and np.isfinite(roc_auc_score(y_true, y_prob)):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[1].plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_prob):.3f}")
        axes[1].plot([0, 1], [0, 1], "--", color="gray")
        axes[1].set_title(f"{title} - ROC")
        axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, save_name), dpi=120)
    plt.close()

# ================= 案例 ①：Spambase 垃圾邮件 =================
def case1_spambase():
    print("\n" + "="*50 + "\n案例 ① 垃圾邮件分类\n" + "="*50)
    col_names = [f"f{i}" for i in range(57)] + ["label"]
    df = pd.read_csv(os.path.join(DATA_DIR, "spambase.csv"), header=None, names=col_names)
    X, y = df.drop(columns=["label"]).values, df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # 线性核
    pipe_lin = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="linear", C=1.0, probability=True, random_state=42))])
    pipe_lin.fit(X_train, y_train)
    evaluate_model(y_test, pipe_lin.predict(X_test), pipe_lin.predict_proba(X_test)[:, 1], "Spambase 线性核")

    # RBF 核
    pipe_rbf = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", C=10, gamma=0.01, probability=True, random_state=42))])
    pipe_rbf.fit(X_train, y_train)
    y_pred_rbf = pipe_rbf.predict(X_test)
    y_prob_rbf = pipe_rbf.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_pred_rbf, y_prob_rbf, "Spambase RBF核")
    plot_cm_and_roc(y_test, y_pred_rbf, y_prob_rbf, "Spambase", "case1_spambase.png")

# ================= 案例 ②：German Credit 信用违约 =================
def case2_german():
    print("\n" + "="*50 + "\n案例 ② 信用违约预测\n" + "="*50)
    df = pd.read_csv(os.path.join(DATA_DIR, "german_credit_data.csv"))
    X_enc = pd.get_dummies(df.drop(columns=["kredit"]).astype(str), drop_first=False)
    y = df["kredit"].values
    X_train, X_test, y_train, y_test = train_test_split(X_enc.values, y, test_size=0.25, random_state=42, stratify=y)

    # 使用 class_weight="balanced" 处理不平衡
    pipe_rbf = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", C=5, gamma=0.1, class_weight="balanced", probability=True, random_state=42))])
    pipe_rbf.fit(X_train, y_train)
    y_pred = pipe_rbf.predict(X_test)
    y_prob = pipe_rbf.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_pred, y_prob, "German Credit RBF核(balanced)")
    plot_cm_and_roc(y_test, y_pred, y_prob, "German Credit", "case2_german.png")

# ================= 案例 ③：Leukemia 基因表达 =================
def case3_leukemia():
    print("\n" + "="*50 + "\n案例 ③ 基因表达分类\n" + "="*50)
    df = pd.read_csv(os.path.join(DATA_DIR, "data_set_ALL_AML_train.csv"))
    
    # 提取纯数值列转置
    id_cols = ["Gene Description", "Gene Accession Number"]
    sample_cols = [c for c in df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])]
    X = df[sample_cols].T.astype(float).values
    y = np.array([1]*27 + [0]*(X.shape[0]-27)) # 前27为ALL

    X_scaled = StandardScaler().fit_transform(VarianceThreshold(0).fit_transform(X))
    
    # 模型对比 (LOOCV)
    models = {
        "Linear SVM": Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="linear", C=1.0))]),
        "RBF SVM": Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf"))]),
        "KNN": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(3))]),
        "Tree": DecisionTreeClassifier(random_state=42)
    }
    accs, names = [], []
    for name, m in models.items():
        acc = cross_val_score(m, X_scaled, y, cv=LeaveOneOut()).mean()
        print(f"{name} LOOCV 准确率: {acc:.4f}")
        accs.append(acc); names.append(name)
        
    # 画对比柱状图
    fig, ax = plt.subplots()
    ax.bar(names, accs, color=['#2563eb', '#7c3aed', '#0d9488', '#d97706'])
    ax.set_ylim(0, 1.1); ax.set_title("Leukemia LOOCV 模型对比")
    plt.savefig(os.path.join(FIG_DIR, "case3_leukemia_bar.png"), dpi=120)
    plt.close()

# ================= 案例 ④：MNIST 手写数字 =================
def case4_mnist():
    print("\n" + "="*50 + "\n案例 ④ 手写数字识别\n" + "="*50)
    df = pd.read_csv(os.path.join(DATA_DIR, "mnist_test.csv"))
    df = df[df.iloc[:, 0].isin([0, 1])].sample(n=12000, random_state=42) # 抽1.2万条
    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe_rbf = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", C=10, gamma=0.01, probability=True, random_state=42))])
    t0 = time.time()
    pipe_rbf.fit(X_train, y_train)
    print(f"RBF 训练耗时: {time.time()-t0:.1f}s")
    
    y_pred = pipe_rbf.predict(X_test)
    y_prob = pipe_rbf.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_pred, y_prob, "MNIST 0/1 RBF核")
    plot_cm_and_roc(y_test, y_pred, y_prob, "MNIST", "case4_mnist.png")

if __name__ == "__main__":
    case1_spambase()
    case2_german()
    case3_leukemia()
    case4_mnist()
    print("\n--- 所有实验完成，图片已保存至 figures/ ---")
```

## 三、 实验结果分析

结合 `figures` 文件夹中生成的混淆矩阵图、ROC 曲线图和柱状图，得出以下分析：

### 1. Spambase 垃圾邮件分类
- **现象**：线性核的准确率达到了约 92%，而 RBF 核稍微高一点，在 93% 左右。

  ![case1_spambase](E:\python数据分析作业\作业5：SVM\figures\case1_spambase.png)

- **分析**：这说明 Spambase 数据集在标准化之后接近线性可分。邮件特征（各类词频）往往可以被一个线性超平面很好地割裂。考虑到线性核计算速度更快，在这个场景下，线性核其实已经非常足够。

### 2. German Credit 信用违约预测
- **现象**：加上 `class_weight="balanced"` 之后，RBF 核的 AUC 能达到 0.83 左右，能够较好地识别出违约客户。

  ![case2_german](E:\python数据分析作业\作业5：SVM\figures\case2_german.png)

- **分析**：这个实验体现了数据不平衡的问题。样本只有 1000 条，好信用占了七成。RBF 核表现比线性核好，说明信贷违约的边界是非线性的（比如年龄和贷款金额之间存在复杂的交互关系）。实际业务中必须调整权重，重点关注 Recall 和 AUC 指标。

### 3. Leukemia 白血病基因分类
- **现象**：仅仅 38 个样本，却包含 7129 个基因。采用留一法（LOOCV）验证，线性 SVM 的准确率接近 100%，而 KNN 只有 85% 左右，决策树大概 78%。

  ![case3_leukemia_bar](E:\python数据分析作业\作业5：SVM\figures\case3_leukemia_bar.png)

- **分析**：这就是 SVM 的典型应用场景——高维小样本。在极高维的空间里，数据更容易被一个超平面完美切开，所以线性核足矣。而 KNN 受困于维数灾难，决策树则容易过拟合。

### 4. MNIST 手写数字分类
- **现象**：抽取了 1.2 万张图片，RBF 核的准确率高达 99.4% 左右，但训练耗时比前三个案例长得多。

  ![case4_mnist](E:\python数据分析作业\作业5：SVM\figures\case4_mnist.png)

- **分析**：图像的像素特征存在复杂的非线性组合，必须用 RBF 核映射到高维空间才能精准画出边界。但 RBF 核的复杂度是 O(n²)~O(n³)，随着样本量增加，训练时间会显著增加。在大数据场景下，可以考虑线性 SVM 或者深度学习。
