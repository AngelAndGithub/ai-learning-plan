#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第138天：模型训练和评估
项目实战学习示例
内容：模型选择、模型训练、模型评估和模型调优
"""

print("=== 第138天：模型训练和评估 ===")

# 1. 模型选择
print("\n1. 模型选择")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

print("模型选择是项目成功的关键")
print("- 线性模型：线性回归、逻辑回归")
print("- 树模型：决策树、随机森林、梯度提升树")
print("- 支持向量机")
print("- K近邻")
print("- 神经网络")

# 2. 数据集准备
print("\n2. 数据集准备")

# 生成示例数据
data = {
    'id': range(1, 101),
    'age': np.random.randint(18, 65, 100),
    'gender': np.random.choice(['Male', 'Female'], 100),
    'income': np.random.randint(30000, 100000, 100),
    'expenses': np.random.randint(10000, 80000, 100),
    'credit_score': np.random.randint(300, 850, 100),
    'default': np.random.choice([0, 1], 100, p=[0.8, 0.2])
}

df = pd.DataFrame(data)

# 数据预处理
le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])

# 特征工程
df['savings'] = df['income'] - df['expenses']
df['debt_ratio'] = df['expenses'] / df['income']

# 特征缩放
scaler = StandardScaler()
df[['age', 'income', 'expenses', 'credit_score', 'savings', 'debt_ratio']] = scaler.fit_transform(
    df[['age', 'income', 'expenses', 'credit_score', 'savings', 'debt_ratio']]
)

# 数据划分
X = df.drop(['id', 'gender', 'default'], axis=1)
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

print(f"训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")
print(f"测试集大小: {len(X_test)}")

# 3. 模型训练
print("\n3. 模型训练")

print("模型训练是模型性能的基础")

# 初始化模型
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier()
}

# 训练模型
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # 计算评估指标
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'model': model
    }
    
    print(f"\n{name} 验证集性能:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

# 4. 模型评估
print("\n4. 模型评估")

print("模型评估是选择最佳模型的依据")

# 比较不同模型的性能
results_df = pd.DataFrame(results).T
print("\n模型性能比较:")
print(results_df[['accuracy', 'precision', 'recall', 'f1']].sort_values('f1', ascending=False))

# 选择最佳模型
best_model_name = results_df['f1'].idxmax()
best_model = results[best_model_name]['model']
print(f"\n最佳模型: {best_model_name}")

# 在测试集上评估最佳模型
y_pred_test = best_model.predict(X_test)
y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]

print("\n最佳模型测试集性能:")
print(f"准确率: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"精确率: {precision_score(y_test, y_pred_test):.4f}")
print(f"召回率: {recall_score(y_test, y_pred_test):.4f}")
print(f"F1分数: {f1_score(y_test, y_pred_test):.4f}")

# 混淆矩阵
print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred_test))

# ROC曲线
print("\nROC曲线:")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 5. 模型调优
print("\n5. 模型调优")

print("模型调优是提高模型性能的重要步骤")

# 示例：使用GridSearchCV调优随机森林
print("\n使用GridSearchCV调优随机森林:")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# 使用最佳参数重新训练模型
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# 评估调优后的模型
y_pred_test_tuned = best_rf.predict(X_test)
y_pred_proba_test_tuned = best_rf.predict_proba(X_test)[:, 1]

print("\n调优后模型测试集性能:")
print(f"准确率: {accuracy_score(y_test, y_pred_test_tuned):.4f}")
print(f"精确率: {precision_score(y_test, y_pred_test_tuned):.4f}")
print(f"召回率: {recall_score(y_test, y_pred_test_tuned):.4f}")
print(f"F1分数: {f1_score(y_test, y_pred_test_tuned):.4f}")

# 6. 特征重要性
print("\n6. 特征重要性")

print("特征重要性分析有助于理解模型")

# 查看随机森林的特征重要性
if hasattr(best_model, 'feature_importances_'):
    feature_importances = best_model.feature_importances_
    feature_names = X.columns
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    print("\n特征重要性:")
    print(importance_df)
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Feature Importance')
    plt.show()

# 7. 模型保存和加载
print("\n7. 模型保存和加载")

print("模型保存和加载是模型部署的基础")

import joblib

# 保存模型
joblib.dump(best_model, 'best_model.pkl')
print("模型已保存为 best_model.pkl")

# 加载模型
loaded_model = joblib.load('best_model.pkl')
print("模型已加载")

# 测试加载的模型
y_pred_loaded = loaded_model.predict(X_test)
print(f"加载模型的准确率: {accuracy_score(y_test, y_pred_loaded):.4f}")

# 8. 模型部署
print("\n8. 模型部署")

print("模型部署是将模型应用到生产环境的过程")
print("- 本地部署")
print("- 云服务部署")
print("- 容器化部署")
print("- API部署")

# 9. 练习
print("\n9. 练习")

# 练习1: 模型选择
print("练习1: 模型选择")
print("- 尝试不同的模型")
print("- 比较模型性能")
print("- 选择最佳模型")

# 练习2: 模型调优
print("\n练习2: 模型调优")
print("- 使用GridSearchCV调优模型")
print("- 尝试不同的参数组合")
print("- 评估调优效果")

# 练习3: 模型评估
print("\n练习3: 模型评估")
print("- 使用不同的评估指标")
print("- 绘制ROC曲线")
print("- 分析混淆矩阵")

# 练习4: 特征重要性
print("\n练习4: 特征重要性")
print("- 分析特征重要性")
print("- 选择重要特征")
print("- 重新训练模型")

print("\n=== 第138天学习示例结束 ===")
