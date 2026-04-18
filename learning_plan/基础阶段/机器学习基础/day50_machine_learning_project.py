#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第50天：机器学习项目实战
机器学习基础学习示例
内容：机器学习项目的完整流程、数据处理、模型训练与评估、部署
"""

print("=== 第50天：机器学习项目实战 ===")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import joblib
import os

# 1. 项目概述
print("\n1. 项目概述")

print("本项目使用乳腺癌数据集进行分类任务")
print("- 数据集：Wisconsin Breast Cancer Dataset")
print("- 任务：二分类，判断肿瘤是良性还是恶性")
print("- 目标：构建一个准确的分类模型")

# 2. 数据获取与探索
print("\n2. 数据获取与探索")

# 加载数据集
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
print(f"数据集形状: {X.shape}, {y.shape}")
print(f"特征名称: {cancer.feature_names[:5]}...")  # 显示前5个特征
print(f"类别名称: {cancer.target_names}")
print(f"类别分布: 良性={np.sum(y == 0)}, 恶性={np.sum(y == 1)}")

# 创建DataFrame进行数据探索
df = pd.DataFrame(X, columns=cancer.feature_names)
df['target'] = y
df['target_name'] = df['target'].map({0: '良性', 1: '恶性'})

# 数据基本信息
print("\n数据基本信息:")
print(df.info())

# 数据统计描述
print("\n数据统计描述:")
print(df.describe())

# 3. 数据预处理
print("\n3. 数据预处理")

# 检查缺失值
print("\n检查缺失值:")
print(df.isnull().sum())

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
print(f"训练集形状: {X_train.shape}, {y_train.shape}")
print(f"测试集形状: {X_test.shape}, {y_test.shape}")

# 4. 模型训练与评估
print("\n4. 模型训练与评估")

# 定义模型
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# 训练和评估模型
results = []
for model_name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # 计算ROC曲线和AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # 存储结果
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': roc_auc
    })
    
    # 打印结果
    print(f"\n{model_name}:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    print("分类报告:")
    print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# 5. 模型比较
print("\n5. 模型比较")

# 转换结果为DataFrame
results_df = pd.DataFrame(results)
print("模型性能比较:")
print(results_df.sort_values('F1 Score', ascending=False))

# 可视化ROC曲线
plt.figure(figsize=(10, 6))
for model_name, model in models.items():
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
print("ROC曲线已保存为 roc_curve.png")

# 6. 模型调优
print("\n6. 模型调优")

print("使用网格搜索调优随机森林模型")

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 创建网格搜索对象
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

# 开始搜索
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳参数进行预测
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)

print(f"\n调优后模型性能:")
print(f"准确率: {accuracy_best:.4f}")
print(f"精确率: {precision_best:.4f}")
print(f"召回率: {recall_best:.4f}")
print(f"F1分数: {f1_best:.4f}")

# 7. 特征重要性分析
print("\n7. 特征重要性分析")

# 获取特征重要性
feature_importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': cancer.feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print("特征重要性排序:")
print(feature_importance_df.head(10))

# 可视化特征重要性
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 特征重要性')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("特征重要性图已保存为 feature_importance.png")

# 8. 模型保存与加载
print("\n8. 模型保存与加载")

# 创建模型保存目录
if not os.path.exists('models'):
    os.makedirs('models')

# 保存模型
joblib.dump(best_model, 'models/best_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
print("模型已保存到 models/ 目录")

# 加载模型
loaded_model = joblib.load('models/best_model.joblib')
loaded_scaler = joblib.load('models/scaler.joblib')
print("模型加载成功")

# 测试加载的模型
test_data = X_test[0].reshape(1, -1)
prediction = loaded_model.predict(test_data)
prediction_proba = loaded_model.predict_proba(test_data)
print(f"测试数据预测结果: {cancer.target_names[prediction[0]]}")
print(f"预测概率: {prediction_proba[0]}")

# 9. 模型部署
print("\n9. 模型部署")

print("创建模型部署文件")

# 创建Flask应用文件
with open('app.py', 'w') as f:
    f.write('''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# 加载模型和标准化器
model = joblib.load('models/best_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# 类别名称
class_names = ['良性', '恶性']

@app.route('/predict', methods=['POST'])
def predict():
    """模型预测API"""
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': '请提供特征数据'}), 400
        
        # 处理数据
        features = data['features']
        if not isinstance(features, list):
            return jsonify({'error': '特征数据必须是列表'}), 400
        
        # 转换为numpy数组
        features = np.array(features).reshape(1, -1)
        
        # 标准化
        scaled_features = scaler.transform(features)
        
        # 预测
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)
        
        # 构建响应
        response = {
            'prediction': class_names[prediction[0]],
            'prediction_index': int(prediction[0]),
            'probabilities': prediction_proba[0].tolist(),
            'class_names': class_names
        }
        
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """健康检查API"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
''')
print("app.py已创建")

# 创建Dockerfile
with open('Dockerfile', 'w') as f:
    f.write('''FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir flask scikit-learn numpy pandas joblib seaborn matplotlib

EXPOSE 5000

CMD ["python", "app.py"]
''')
print("Dockerfile已创建")

# 创建docker-compose.yml
with open('docker-compose.yml', 'w') as f:
    f.write('''version: '3'
services:
  breast-cancer-classifier:
    build: .
    ports:
      - "5000:5000"
    restart: always
''')
print("docker-compose.yml已创建")

# 10. 项目总结
print("\n10. 项目总结")

print("本项目成功完成了乳腺癌分类任务")
print("- 数据探索：了解了数据集的基本情况")
print("- 数据预处理：进行了数据标准化")
print("- 模型训练：训练了多个分类模型")
print("- 模型评估：评估了模型性能")
print("- 模型调优：使用网格搜索调优了随机森林模型")
print("- 特征重要性：分析了特征的重要性")
print("- 模型部署：创建了模型部署文件")

print("\n项目成果:")
print(f"最佳模型: 随机森林")
print(f"最佳F1分数: {f1_best:.4f}")
print(f"最佳准确率: {accuracy_best:.4f}")

# 11. 练习
print("\n11. 练习")

# 练习1: 扩展项目
print("练习1: 扩展项目")
print("- 使用其他数据集进行分类任务")
print("- 尝试更多的模型和调优方法")

# 练习2: 模型解释
print("\n练习2: 模型解释")
print("- 使用SHAP或LIME解释模型的预测")
print("- 分析模型的决策过程")

# 练习3: 模型监控
print("\n练习3: 模型监控")
print("- 实现模型监控功能")
print("- 监控模型性能和数据漂移")

# 练习4: 部署到云服务
print("\n练习4: 部署到云服务")
print("- 将模型部署到AWS或Azure等云服务")
print("- 测试云部署的性能")

# 练习5: 构建完整的MLOps流程
print("\n练习5: 构建完整的MLOps流程")
print("- 实现自动化的模型训练和部署")
print("- 构建CI/CD pipeline")

print("\n=== 第50天学习示例结束 ===")
