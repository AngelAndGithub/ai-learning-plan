import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston, load_iris, make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 线性回归示例
def linear_regression_example():
    """线性回归示例"""
    print("=== 线性回归示例 ===")
    
    # 生成回归数据集
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    
    # 数据可视化
    plt.scatter(X, y)
    plt.title("线性回归数据集")
    plt.xlabel("特征")
    plt.ylabel("目标")
    plt.savefig('linear_regression_data.png')
    plt.close()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # 可视化预测结果
    plt.scatter(X_test, y_test, color='blue', label='真实值')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='预测值')
    plt.title("线性回归预测结果")
    plt.xlabel("特征")
    plt.ylabel("目标")
    plt.legend()
    plt.savefig('linear_regression_prediction.png')
    plt.close()
    
    return model

# 2. 逻辑回归示例
def logistic_regression_example():
    """逻辑回归示例"""
    print("\n=== 逻辑回归示例 ===")
    
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data[:, :2]  # 只使用前两个特征
    y = (iris.target == 0).astype(int)  # 二分类问题：是否为山鸢尾
    
    # 数据可视化
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='山鸢尾')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='非山鸢尾')
    plt.title("逻辑回归数据集")
    plt.xlabel("花萼长度")
    plt.ylabel("花萼宽度")
    plt.legend()
    plt.savefig('logistic_regression_data.png')
    plt.close()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # 可视化决策边界
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='山鸢尾')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='非山鸢尾')
    plt.title("逻辑回归决策边界")
    plt.xlabel("花萼长度")
    plt.ylabel("花萼宽度")
    plt.legend()
    plt.savefig('logistic_regression_boundary.png')
    plt.close()
    
    return model

# 3. 决策树示例
def decision_tree_example():
    """决策树示例"""
    print("\n=== 决策树示例 ===")
    
    # 生成分类数据集
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    
    # 数据可视化
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='类别 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='类别 1')
    plt.title("决策树数据集")
    plt.xlabel("特征 1")
    plt.ylabel("特征 2")
    plt.legend()
    plt.savefig('decision_tree_data.png')
    plt.close()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    
    # 可视化决策边界
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='类别 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='类别 1')
    plt.title("决策树决策边界")
    plt.xlabel("特征 1")
    plt.ylabel("特征 2")
    plt.legend()
    plt.savefig('decision_tree_boundary.png')
    plt.close()
    
    return model

# 4. 随机森林示例
def random_forest_example():
    """随机森林示例"""
    print("\n=== 随机森林示例 ===")
    
    # 生成分类数据集
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    
    # 可视化决策边界
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='类别 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='类别 1')
    plt.title("随机森林决策边界")
    plt.xlabel("特征 1")
    plt.ylabel("特征 2")
    plt.legend()
    plt.savefig('random_forest_boundary.png')
    plt.close()
    
    # 特征重要性
    feature_importance = model.feature_importances_
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title("特征重要性")
    plt.xlabel("特征")
    plt.ylabel("重要性")
    plt.savefig('random_forest_feature_importance.png')
    plt.close()
    
    return model

# 5. 支持向量机示例
def svm_example():
    """支持向量机示例"""
    print("\n=== 支持向量机示例 ===")
    
    # 生成分类数据集
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    # 线性核
    model_linear = SVC(kernel='linear', random_state=42)
    model_linear.fit(X_train_scaled, y_train)
    
    # RBF 核
    model_rbf = SVC(kernel='rbf', random_state=42)
    model_rbf.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred_linear = model_linear.predict(X_test_scaled)
    y_pred_rbf = model_rbf.predict(X_test_scaled)
    
    # 评估
    print("线性核:")
    print(f"准确率: {accuracy_score(y_test, y_pred_linear):.4f}")
    print(f"F1 分数: {f1_score(y_test, y_pred_linear):.4f}")
    
    print("\nRBF 核:")
    print(f"准确率: {accuracy_score(y_test, y_pred_rbf):.4f}")
    print(f"F1 分数: {f1_score(y_test, y_pred_rbf):.4f}")
    
    # 可视化决策边界
    def plot_decision_boundary(model, X, y, title):
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='类别 0')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='类别 1')
        plt.title(title)
        plt.xlabel("特征 1")
        plt.ylabel("特征 2")
        plt.legend()
        plt.savefig(f'svm_{title.lower().replace(" ", "_")}.png')
        plt.close()
    
    plot_decision_boundary(model_linear, X_train_scaled, y_train, "SVM 线性核决策边界")
    plot_decision_boundary(model_rbf, X_train_scaled, y_train, "SVM RBF 核决策边界")
    
    return model_linear, model_rbf

# 6. 模型调优示例
def model_tuning_example():
    """模型调优示例"""
    print("\n=== 模型调优示例 ===")
    
    # 生成分类数据集
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }
    
    # 网格搜索
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳 F1 分数: {grid_search.best_score_:.4f}")
    
    # 使用最佳参数
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(f"测试集 F1 分数: {f1_score(y_test, y_pred):.4f}")
    
    return best_model

# 7. 完整的监督学习流程示例
def complete_workflow_example():
    """完整的监督学习流程示例"""
    print("\n=== 完整的监督学习流程示例 ===")
    
    # 加载波士顿房价数据集
    boston = load_boston()
    X, y = boston.data, boston.target
    
    print(f"数据集形状: X={X.shape}, y={y.shape}")
    print(f"特征名称: {boston.feature_names}")
    
    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 模型选择与训练
    models = {
        '线性回归': LinearRegression(),
        '决策树回归': DecisionTreeRegressor(max_depth=5, random_state=42),
        '随机森林回归': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        print(f"{name} 交叉验证 R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 训练并评估
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        results[name] = {
            'r2': r2,
            'mse': mse,
            'rmse': rmse
        }
        
        print(f"{name} 测试集 R²: {r2:.4f}")
        print(f"{name} 测试集 MSE: {mse:.4f}")
        print(f"{name} 测试集 RMSE: {rmse:.4f}")
        print()
    
    # 可视化预测结果
    plt.figure(figsize=(12, 4))
    for i, (name, model) in enumerate(models.items(), 1):
        plt.subplot(1, 3, i)
        y_pred = model.predict(X_test)
        plt.scatter(y_test, y_pred)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.title(f"{name} 预测 vs 真实")
        plt.xlabel("真实值")
        plt.ylabel("预测值")
    plt.tight_layout()
    plt.savefig('supervised_learning_comparison.png')
    plt.close()
    
    return results

if __name__ == "__main__":
    # 运行所有示例
    linear_regression_example()
    logistic_regression_example()
    decision_tree_example()
    random_forest_example()
    svm_example()
    model_tuning_example()
    complete_workflow_example()
    
    print("\n所有示例运行完成！")