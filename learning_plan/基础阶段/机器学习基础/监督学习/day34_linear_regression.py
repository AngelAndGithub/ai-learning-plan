#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第34天：线性回归
机器学习基础学习示例
内容：线性回归的深入理解、多元回归和正则化
"""

print("=== 第34天：线性回归 ===")

# 1. 简单线性回归
print("\n1. 简单线性回归")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 生成简单线性回归数据
np.random.seed(42)
x = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * x + 1 + np.random.randn(100, 1) * 2

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"简单线性回归评估:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"R² 评分: {r2:.4f}")
print(f"系数: {model.coef_[0][0]:.4f}")
print(f"截距: {model.intercept_[0]:.4f}")

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, label='训练数据', alpha=0.6)
plt.scatter(X_test, y_test, label='测试数据', alpha=0.6, color='green')
plt.plot(x, model.predict(x), 'r-', label='线性回归拟合')
plt.xlabel('X')
plt.ylabel('y')
plt.title('简单线性回归')
plt.legend()
plt.show()

# 2. 多元线性回归
print("\n2. 多元线性回归")

from sklearn.datasets import load_diabetes

# 加载糖尿病数据集
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

print(f"特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")
print(f"特征名称: {diabetes.feature_names}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"多元线性回归评估:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"R² 评分: {r2:.4f}")

# 显示系数
print("\n特征系数:")
for feature, coef in zip(diabetes.feature_names, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"截距: {model.intercept_:.4f}")

# 3. 多项式回归
print("\n3. 多项式回归")

from sklearn.preprocessing import PolynomialFeatures

# 生成非线性数据
np.random.seed(42)
x = np.linspace(-3, 3, 100).reshape(-1, 1)
y = x**3 + x**2 + x + 1 + np.random.randn(100, 1) * 5

# 多项式特征转换
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(x)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"多项式回归评估:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"R² 评分: {r2:.4f}")

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label='原始数据', alpha=0.6)

# 排序以获得平滑曲线
sort_indices = np.argsort(x.flatten())
x_sorted = x[sort_indices]
X_poly_sorted = X_poly[sort_indices]
y_pred_sorted = model.predict(X_poly_sorted)

plt.plot(x_sorted, y_pred_sorted, 'r-', label='多项式回归拟合')
plt.xlabel('X')
plt.ylabel('y')
plt.title('多项式回归 (degree=3)')
plt.legend()
plt.show()

# 4. 正则化线性回归
print("\n4. 正则化线性回归")

from sklearn.linear_model import Ridge, Lasso

# 使用糖尿病数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge回归（L2正则化）
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Lasso回归（L1正则化）
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"Ridge回归评估:")
print(f"均方误差 (MSE): {mse_ridge:.4f}")
print(f"R² 评分: {r2_ridge:.4f}")

print(f"\nLasso回归评估:")
print(f"均方误差 (MSE): {mse_lasso:.4f}")
print(f"R² 评分: {r2_lasso:.4f}")

# 比较系数
print("\n系数比较:")
print("特征\t\tLinear\t\tRidge\t\tLasso")
for feature, coef_linear, coef_ridge, coef_lasso in zip(
    diabetes.feature_names, model.coef_, ridge.coef_, lasso.coef_
):
    print(f"{feature}\t\t{coef_linear:.4f}\t\t{coef_ridge:.4f}\t\t{coef_lasso:.4f}")

# 5. 交叉验证
print("\n5. 交叉验证")

from sklearn.model_selection import cross_val_score

# 对线性回归进行交叉验证
model = LinearRegression()
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"5折交叉验证 R² 评分:")
print(cv_scores)
print(f"平均 R² 评分: {cv_scores.mean():.4f}")
print(f"标准差: {cv_scores.std():.4f}")

# 6. 模型选择
print("\n6. 模型选择")

from sklearn.model_selection import GridSearchCV

# 为Ridge回归寻找最佳alpha值
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
grid_search.fit(X, y)

print(f"Ridge回归最佳参数:")
print(grid_search.best_params_)
print(f"最佳交叉验证评分: {grid_search.best_score_:.4f}")

# 7. 特征选择
print("\n7. 特征选择")

from sklearn.feature_selection import SelectKBest, f_regression

# 选择K个最佳特征
selector = SelectKBest(f_regression, k=5)
X_selected = selector.fit_transform(X, y)

print(f"原始特征数量: {X.shape[1]}")
print(f"选择后特征数量: {X_selected.shape[1]}")
print(f"选择的特征索引: {selector.get_support(indices=True)}")
print(f"选择的特征名称: {[diabetes.feature_names[i] for i in selector.get_support(indices=True)]}")

# 8. 残差分析
print("\n8. 残差分析")

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 计算残差
residuals = y_test - y_pred

# 可视化残差
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差图')
plt.show()

# 9. 异常值检测
print("\n9. 异常值检测")

# 使用Z-score方法检测异常值
from scipy import stats

# 计算Z-score
z_scores = np.abs(stats.zscore(y))
threshold = 3
outliers = np.where(z_scores > threshold)

print(f"异常值索引: {outliers[0]}")
print(f"异常值数量: {len(outliers[0])}")
print(f"异常值: {y[outliers]}")

# 10. 练习
print("\n10. 练习")

# 练习1: 不同多项式阶数的影响
print("练习1: 不同多项式阶数的影响")

# 生成数据
np.random.seed(42)
x = np.linspace(-3, 3, 100).reshape(-1, 1)
y = x**3 + x**2 + x + 1 + np.random.randn(100, 1) * 5

# 尝试不同的多项式阶数
degrees = [1, 2, 3, 10]

plt.figure(figsize=(12, 8))

for i, degree in enumerate(degrees, 1):
    # 多项式特征转换
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(x)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"多项式阶数 {degree}: MSE = {mse:.4f}, R² = {r2:.4f}")
    
    # 可视化
    plt.subplot(2, 2, i)
    plt.scatter(x, y, label='原始数据', alpha=0.6)
    
    # 排序以获得平滑曲线
    sort_indices = np.argsort(x.flatten())
    x_sorted = x[sort_indices]
    X_poly_sorted = X_poly[sort_indices]
    y_pred_sorted = model.predict(X_poly_sorted)
    
    plt.plot(x_sorted, y_pred_sorted, 'r-', label=f'阶数={degree}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'多项式回归 (阶数={degree})')
    plt.legend()

plt.tight_layout()
plt.show()

print("\n=== 第34天学习示例结束 ===")
