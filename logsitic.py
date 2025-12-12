import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =====================
# 请在这里替换为你的数据集实际路径
# 方式1：绝对路径（示例）
# fileText = r"C:\Users\E507\Desktop\logistic\horseColicTest.txt"
# fileTrain = r"C:\Users\E507\Desktop\logistic\horseColicTraining.txt"
# 方式2：相对路径（推荐，将数据集和py文件放同一文件夹）
fileText = "horseColicTest.txt"
fileTrain = "horseColicTraining.txt"

# =====================
# 1. 路径检测函数
# =====================
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误：文件 {file_path} 不存在，请检查路径！")
        return False
    print(f"✅ 成功找到文件：{file_path}")
    return True

# =====================
# 2. 数据读取函数
# =====================
def load_dataset(filename):
    try:
        data = np.loadtxt(filename)
        X = data[:, :-1]  # 特征
        y = data[:, -1]   # 标签
        print(f"✅ 数据读取完成，特征维度：{X.shape}，标签数量：{y.shape[0]}")
        return X, y
    except Exception as e:
        print(f"❌ 数据读取失败：{str(e)}")
        return None, None

# =====================
# 3. 缺失值处理函数（修正版）
# =====================
def replace_nan_with_mean(X):
    X_processed = X.copy()  # 避免修改原数据
    for i in range(X_processed.shape[1]):
        col = X_processed[:, i]
        missing_mask = col == 0  # 标记缺失值（0代表缺失）
        if np.sum(missing_mask) > 0:
            valid = col[~missing_mask]
            if len(valid) > 0:
                mean_val = np.mean(valid)
                col[missing_mask] = mean_val
                X_processed[:, i] = col
            else:
                col[missing_mask] = 0  # 整列缺失则填充0
    print(f"✅ 缺失值处理完成，处理后特征维度：{X_processed.shape}")
    return X_processed

# =====================
# 4. 主流程
# =====================
if __name__ == "__main__":
    # 第一步：检测文件路径
    print("===== 检测文件路径 =====")
    if not check_file_exists(fileTrain) or not check_file_exists(fileText):
        exit()

    # 第二步：读取数据
    print("\n===== 读取训练集 =====")
    X_train, y_train = load_dataset(fileTrain)
    if X_train is None:
        exit()

    print("\n===== 读取测试集 =====")
    X_test, y_test = load_dataset(fileText)
    if X_test is None:
        exit()

    # 第三步：处理缺失值
    print("\n===== 处理训练集缺失值 =====")
    X_train_processed = replace_nan_with_mean(X_train)
    print("\n===== 处理测试集缺失值 =====")
    X_test_processed = replace_nan_with_mean(X_test)

    # 第四步：训练逻辑回归模型（调优版）
    print("\n===== 训练逻辑回归模型 =====")
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=2000,
        solver='liblinear',  # 对小数据集更稳定
        C=1.0  # 正则化强度，可根据需求调整
    )
    lr_model.fit(X_train_processed, y_train)
    print(f"✅ 模型训练完成")
    print(f"模型系数（权重）：{lr_model.coef_}")
    print(f"模型截距（偏置）：{lr_model.intercept_}")

    # 第五步：预测与评估
    print("\n===== 测试集预测 =====")
    y_pred = lr_model.predict(X_test_processed)
    y_pred_proba = lr_model.predict_proba(X_test_processed)
    print(f"前10个预测结果：{y_pred[:10]}")
    print(f"前10个真实标签：{y_test[:10]}")

    print("\n===== 模型评估 =====")
    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train, lr_model.predict(X_train_processed))
    print(f"测试集准确率：{test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"训练集准确率：{train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

    # 各类别准确率
    unique_classes = np.unique(y_test)
    for cls in unique_classes:
        cls_indices = y_test == cls
        cls_accuracy = accuracy_score(y_test[cls_indices], y_pred[cls_indices])
        print(f"类别 {cls} 的准确率：{cls_accuracy:.4f}")