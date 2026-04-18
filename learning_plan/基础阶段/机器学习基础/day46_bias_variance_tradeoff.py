#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第46天：偏差-方差权衡
机器学习基础学习示例
内容：偏差-方差权衡的基本概念、过拟合与欠拟合、模型复杂度与性能的关系
"""

print("=== 第46天：偏差-方差权衡 ===")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 1. 偏差-方差权衡概述
print("\n1. 偏差-方差权衡概述")

print("偏差-方差权衡是机器学习中的核心概念")
print("- 偏差：模型预测值与真实值之间的系统误差")
print("- 方差：模型预测值的变化范围")
print("- 总误差 = 偏差² + 方差 + 噪声")
print("- 目标：找到偏差和方差的平衡点")

# 2. 生成合成数据集
print("\n2. 生成合成数据集")

# 生成非线性数据
def generate_data(n_samples=100, noise=0.1):
    """生成非线性数据集"""
    np.random.seed(42)
    X = np.linspace(0, 1, n_samples)
    y = np.sin(2 * np.pi * X) + np.random.normal(0, noise, n_samples)
    return X.reshape(-1, 1), y

X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"数据集形状: {X.shape}, {y.shape}")
print(f"训练集形状: {X_train.shape}, {y_train.shape}")
print(f"测试集形状: {X_test.shape}, {y_test.shape}")

# 可视化数据
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='数据点')
plt.plot(np.linspace(0, 1, 100), np.sin(2 * np.pi * np.linspace(0, 1, 100)), 'r-', label='真实函数')
plt.title('合成数据集')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.savefig('synthetic_data.png')
print("合成数据集已保存为 synthetic_data.png")

# 3. 不同复杂度模型的拟合
print("\n3. 不同复杂度模型的拟合")

# 定义不同复杂度的模型
def train_models(X_train, y_train, X_test, y_test):
    """训练不同复杂度的模型"""
    models = []
    train_errors = []
    test_errors = []
    complexities = []
    
    # 线性模型（低复杂度）
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=1)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    models.append('线性模型')
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))
    complexities.append(1)
    
    # 多项式模型（中等复杂度）
    for degree in [2, 3, 5]:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        models.append(f'多项式模型 (degree={degree})')
        train_errors.append(mean_squared_error(y_train, y_train_pred))
        test_errors.append(mean_squared_error(y_test, y_test_pred))
        complexities.append(degree)
    
    # 多项式模型（高复杂度）
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=15)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    models.append('多项式模型 (degree=15)')
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))
    complexities.append(15)
    
    # 决策树（高复杂度）
    model = DecisionTreeRegressor(max_depth=None, random_state=42)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    models.append('决策树 (max_depth=None)')
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))
    complexities.append('DT')
    
    return models, train_errors, test_errors, complexities

# 训练模型
models, train_errors, test_errors, complexities = train_models(X_train, y_train, X_test, y_test)

# 打印结果
print("不同模型的训练误差和测试误差:")
for i, model in enumerate(models):
    print(f"{model}: 训练误差 = {train_errors[i]:.4f}, 测试误差 = {test_errors[i]:.4f}")

# 4. 偏差-方差权衡的可视化
print("\n4. 偏差-方差权衡的可视化")

# 绘制不同复杂度模型的拟合曲线
plt.figure(figsize=(15, 10))
for i, degree in enumerate([1, 2, 3, 5, 15]):
    ax = plt.subplot(2, 3, i + 1)
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    model.fit(X_train, y_train)
    X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    ax.scatter(X_train, y_train, label='训练数据')
    ax.plot(X_plot, y_plot, 'r-', label=f'拟合曲线 (degree={degree})')
    ax.plot(np.linspace(0, 1, 100), np.sin(2 * np.pi * np.linspace(0, 1, 100)), 'g--', label='真实函数')
    ax.set_title(f'多项式模型 (degree={degree})')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
plt.tight_layout()
plt.savefig('model_fitting.png')
print("模型拟合结果已保存为 model_fitting.png")

# 绘制训练误差和测试误差曲线
plt.figure(figsize=(10, 6))
plt.plot(range(len(models)), train_errors, 'o-', label='训练误差')
plt.plot(range(len(models)), test_errors, 's-', label='测试误差')
plt.xticks(range(len(models)), models, rotation=45, ha='right')
plt.title('不同模型的训练误差和测试误差')
plt.xlabel('模型')
plt.ylabel('均方误差')
plt.legend()
plt.tight_layout()
plt.savefig('error_curve.png')
print("误差曲线已保存为 error_curve.png")

# 5. 过拟合与欠拟合
print("\n5. 过拟合与欠拟合")

print("过拟合和欠拟合是机器学习中的常见问题")
print("- 欠拟合：模型无法捕捉数据的基本模式")
print("- 过拟合：模型过度拟合训练数据，泛化能力差")

# 绘制过拟合和欠拟合的示意图
plt.figure(figsize=(12, 6))

# 欠拟合
ax1 = plt.subplot(1, 2, 1)
model = Pipeline([
    ('poly', PolynomialFeatures(degree=1)),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
model.fit(X_train, y_train)
X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
y_plot = model.predict(X_plot)
ax1.scatter(X, y, label='数据点')
ax1.plot(X_plot, y_plot, 'r-', label='拟合曲线')
ax1.plot(np.linspace(0, 1, 100), np.sin(2 * np.pi * np.linspace(0, 1, 100)), 'g--', label='真实函数')
ax1.set_title('欠拟合')
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.legend()

# 过拟合
ax2 = plt.subplot(1, 2, 2)
model = Pipeline([
    ('poly', PolynomialFeatures(degree=15)),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
model.fit(X_train, y_train)
X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
y_plot = model.predict(X_plot)
ax2.scatter(X, y, label='数据点')
ax2.plot(X_plot, y_plot, 'r-', label='拟合曲线')
ax2.plot(np.linspace(0, 1, 100), np.sin(2 * np.pi * np.linspace(0, 1, 100)), 'g--', label='真实函数')
ax2.set_title('过拟合')
ax2.set_xlabel('X')
ax2.set_ylabel('y')
ax2.legend()

plt.tight_layout()
plt.savefig('overfitting_underfitting.png')
print("过拟合和欠拟合示意图已保存为 overfitting_underfitting.png")

# 6. 正则化方法
print("\n6. 正则化方法")

print("正则化是解决过拟合的有效方法")
print("- L1正则化（Lasso）：产生稀疏解")
print("- L2正则化（Ridge）：防止过拟合")

# 测试正则化效果
def test_regularization(X_train, y_train, X_test, y_test):
    """测试不同正则化方法的效果"""
    models = []
    train_errors = []
    test_errors = []
    
    # 普通线性回归
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=15)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    models.append('普通线性回归')
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))
    
    # Ridge回归
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=15)),
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=0.1))
    ])
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    models.append('Ridge回归 (alpha=0.1)')
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))
    
    # Lasso回归
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=15)),
        ('scaler', StandardScaler()),
        ('regressor', Lasso(alpha=0.01))
    ])
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    models.append('Lasso回归 (alpha=0.01)')
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))
    
    return models, train_errors, test_errors

# 测试正则化效果
reg_models, reg_train_errors, reg_test_errors = test_regularization(X_train, y_train, X_test, y_test)

# 打印结果
print("正则化方法的效果:")
for i, model in enumerate(reg_models):
    print(f"{model}: 训练误差 = {reg_train_errors[i]:.4f}, 测试误差 = {reg_test_errors[i]:.4f}")

# 7. 模型复杂度与性能的关系
print("\n7. 模型复杂度与性能的关系")

# 绘制学习曲线
def plot_learning_curve(estimator, title, X, y, cv=5):
    """绘制学习曲线"""
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("训练样本数")
    plt.ylabel("得分")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="训练得分")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="交叉验证得分")
    plt.legend(loc="best")
    return plt

# 绘制不同模型的学习曲线
estimators = [
    ('线性模型', Pipeline([
        ('poly', PolynomialFeatures(degree=1)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])),
    ('多项式模型 (degree=3)', Pipeline([
        ('poly', PolynomialFeatures(degree=3)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])),
    ('多项式模型 (degree=15)', Pipeline([
        ('poly', PolynomialFeatures(degree=15)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])),
    ('Ridge回归', Pipeline([
        ('poly', PolynomialFeatures(degree=15)),
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=0.1))
    ]))
]

for name, estimator in estimators:
    plot_learning_curve(estimator, f"学习曲线: {name}", X, y)
    plt.savefig(f'learning_curve_{name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").lower()}.png')
    plt.close()
    print(f"{name}的学习曲线已保存")

# 8. 偏差-方差权衡的实际应用
print("\n8. 偏差-方差权衡的实际应用")

print("在实际应用中如何平衡偏差和方差:")
print("- 选择合适的模型复杂度")
print("- 使用正则化方法")
print("- 增加训练数据")
print("- 使用集成方法")
print("- 交叉验证评估模型性能")

# 9. 集成方法
print("\n9. 集成方法")

print("集成方法是平衡偏差和方差的有效方法")
print("- 随机森林：降低方差")
print("- 梯度提升：降低偏差")

# 测试集成方法
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_error = mean_squared_error(y_train, y_train_pred)
test_error = mean_squared_error(y_test, y_test_pred)
print(f"随机森林：训练误差 = {train_error:.4f}, 测试误差 = {test_error:.4f}")

# 10. 练习
print("\n10. 练习")

# 练习1: 实现偏差-方差分解
print("练习1: 实现偏差-方差分解")
print("- 实现偏差-方差分解的计算")
print("- 分析不同模型的偏差和方差")

# 练习2: 模型复杂度调优
print("\n练习2: 模型复杂度调优")
print("- 对决策树模型进行复杂度调优")
print("- 分析不同max_depth对模型性能的影响")

# 练习3: 正则化参数调优
print("\n练习3: 正则化参数调优")
print("- 对Ridge和Lasso回归进行参数调优")
print("- 分析不同alpha值对模型性能的影响")

# 练习4: 集成方法的应用
print("\n练习4: 集成方法的应用")
print("- 比较随机森林和梯度提升的性能")
print("- 分析集成方法如何平衡偏差和方差")

# 练习5: 实际数据集的偏差-方差分析
print("\n练习5: 实际数据集的偏差-方差分析")
print("- 选择一个实际数据集")
print("- 分析不同模型的偏差-方差权衡")
print("- 选择最佳模型")

print("\n=== 第46天学习示例结束 ===")
