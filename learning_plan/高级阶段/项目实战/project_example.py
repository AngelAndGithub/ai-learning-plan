import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import json
from flask import Flask, request, jsonify
import os

# 1. 项目初始化
def project_initialization():
    """项目初始化"""
    print("=== 项目初始化 ===")
    print("项目名称: 鸢尾花分类")
    print("项目目标: 使用机器学习模型对鸢尾花进行分类")
    print("技术栈: Python, scikit-learn, Flask, Docker")
    print("=== 项目初始化完成 ===")
    return "项目初始化完成"

# 2. 数据收集与探索
def data_collection_exploration():
    """数据收集与探索"""
    print("\n=== 数据收集与探索 ===")
    
    # 加载数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # 创建DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['species'] = [target_names[i] for i in y]
    
    # 数据概览
    print("数据形状:", df.shape)
    print("\n数据前5行:")
    print(df.head())
    print("\n数据统计信息:")
    print(df.describe())
    print("\n类别分布:")
    print(df['species'].value_counts())
    
    # 数据可视化
    # 特征相关性热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('特征相关性热力图')
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    # 特征分布
    plt.figure(figsize=(12, 10))
    for i, feature in enumerate(feature_names):
        plt.subplot(2, 2, i+1)
        sns.boxplot(x='species', y=feature, data=df)
        plt.title(f'{feature} 分布')
    plt.tight_layout()
    plt.savefig('feature_distribution.png')
    plt.close()
    
    print("\n数据探索完成，生成了可视化图表")
    return df, feature_names, target_names

# 3. 数据预处理
def data_preprocessing(df):
    """数据预处理"""
    print("\n=== 数据预处理 ===")
    
    # 提取特征和标签
    X = df.drop(['target', 'species'], axis=1).values
    y = df['target'].values
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 保存scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("数据预处理完成，保存了scaler")
    return X_train, X_test, y_train, y_test, scaler

# 4. 模型训练与调优
def model_training(X_train, y_train, X_test, y_test):
    """模型训练与调优"""
    print("\n=== 模型训练与调优 ===")
    
    # 定义模型
    rf = RandomForestClassifier()
    
    # 超参数网格
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3]
    }
    
    # 网格搜索
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # 最佳参数
    print("最佳参数:", grid_search.best_params_)
    
    # 最佳模型
    best_model = grid_search.best_estimator_
    
    # 模型评估
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.4f}")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵:")
    print(cm)
    
    # 保存模型
    with open('iris_classifier.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    print("\n模型训练完成，保存了最佳模型")
    return best_model, accuracy

# 5. API开发
def api_development():
    """API开发"""
    print("\n=== API开发 ===")
    
    # 创建Flask应用
    app = Flask(__name__)
    
    # 加载模型和scaler
    with open('iris_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # 类别映射
    target_names = ['setosa', 'versicolor', 'virginica']
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            # 获取请求数据
            data = request.json
            features = data['features']
            
            # 数据预处理
            features_scaled = scaler.transform([features])
            
            # 预测
            predictions = model.predict(features_scaled)
            class_idx = predictions[0]
            class_name = target_names[class_idx]
            
            # 预测概率
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = float(max(probabilities))
            
            # 返回结果
            return jsonify({
                'class': class_name,
                'confidence': confidence,
                'probabilities': {
                    'setosa': float(probabilities[0]),
                    'versicolor': float(probabilities[1]),
                    'virginica': float(probabilities[2])
                }
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok'})
    
    print("API开发完成，创建了/predict和/health端点")
    return app

# 6. Docker配置
def docker_configuration():
    """Docker配置"""
    print("\n=== Docker配置 ===")
    
    # 生成Dockerfile
    dockerfile_content = """
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
"""
    
    # 生成requirements.txt
    requirements_content = """
Flask
scikit-learn
numpy
pandas
matplotlib
seaborn
"""
    
    # 生成app.py
    app_py_content = """
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# 加载模型和scaler
with open('iris_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 类别映射
target_names = ['setosa', 'versicolor', 'virginica']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取请求数据
        data = request.json
        features = data['features']
        
        # 数据预处理
        features_scaled = scaler.transform([features])
        
        # 预测
        predictions = model.predict(features_scaled)
        class_idx = predictions[0]
        class_name = target_names[class_idx]
        
        # 预测概率
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(max(probabilities))
        
        # 返回结果
        return jsonify({
            'class': class_name,
            'confidence': confidence,
            'probabilities': {
                'setosa': float(probabilities[0]),
                'versicolor': float(probabilities[1]),
                'virginica': float(probabilities[2])
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
"""
    
    # 写入文件
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    
    with open('app.py', 'w') as f:
        f.write(app_py_content)
    
    print("Docker配置完成，生成了Dockerfile、requirements.txt和app.py")
    return "Docker配置完成"

# 7. 项目文档
def project_documentation():
    """项目文档"""
    print("\n=== 项目文档 ===")
    
    # 生成README.md
    readme_content = """
# 鸢尾花分类项目

## 项目简介

这是一个使用机器学习模型对鸢尾花进行分类的项目。项目使用随机森林算法对鸢尾花的四个特征（萼片长度、萼片宽度、花瓣长度、花瓣宽度）进行分析，从而预测鸢尾花的品种（setosa、versicolor、virginica）。

## 技术栈

- Python 3.8+
- scikit-learn
- Flask
- Docker
- Pandas
- Matplotlib
- Seaborn

## 项目结构

```
├── app.py              # Flask API应用
├── Dockerfile          # Docker配置文件
├── requirements.txt    # Python依赖
├── iris_classifier.pkl # 训练好的模型
├── scaler.pkl          # 数据标准化器
├── correlation_heatmap.png # 特征相关性热力图
├── feature_distribution.png # 特征分布箱线图
└── project_example.py  # 项目主脚本
```

## 快速开始

### 1. 环境设置

```bash
# 安装依赖
pip install -r requirements.txt

# 运行项目主脚本
python project_example.py
```

### 2. 运行API

```bash
# 直接运行
python app.py

# 或使用Docker
docker build -t iris-classifier .
docker run -p 5000:5000 iris-classifier
```

### 3. API使用

#### 健康检查

```bash
curl http://localhost:5000/health
```

#### 预测

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## 模型性能

- 准确率: 1.0000
- 分类报告:
  ```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
  ```

## 项目流程

1. **数据收集与探索**：加载鸢尾花数据集，进行数据探索和可视化。
2. **数据预处理**：数据标准化，分割训练集和测试集。
3. **模型训练与调优**：使用网格搜索选择最佳超参数，训练随机森林模型。
4. **模型评估**：评估模型性能，生成分类报告和混淆矩阵。
5. **API开发**：创建Flask API，提供预测和健康检查端点。
6. **Docker配置**：创建Dockerfile和相关配置文件，支持容器化部署。

## 未来改进

- 添加更多特征工程
- 尝试其他机器学习算法
- 部署到云平台
- 添加监控和日志
- 实现CI/CD流程
"""
    
    # 写入文件
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("项目文档完成，生成了README.md")
    return "项目文档完成"

# 8. 项目测试
def project_testing():
    """项目测试"""
    print("\n=== 项目测试 ===")
    
    # 加载模型和scaler
    with open('iris_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # 测试数据
    test_data = [
        [5.1, 3.5, 1.4, 0.2],  # setosa
        [6.2, 2.9, 4.3, 1.3],  # versicolor
        [7.3, 2.9, 6.3, 1.8]   # virginica
    ]
    
    # 目标标签
    target_names = ['setosa', 'versicolor', 'virginica']
    
    print("测试预测结果:")
    for i, features in enumerate(test_data):
        # 数据预处理
        features_scaled = scaler.transform([features])
        
        # 预测
        prediction = model.predict(features_scaled)
        class_idx = prediction[0]
        class_name = target_names[class_idx]
        
        # 预测概率
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = max(probabilities)
        
        print(f"测试样本 {i+1}: {features}")
        print(f"预测结果: {class_name}")
        print(f"置信度: {confidence:.4f}")
        print(f"概率分布: setosa={probabilities[0]:.4f}, versicolor={probabilities[1]:.4f}, virginica={probabilities[2]:.4f}")
        print()
    
    print("项目测试完成")
    return "项目测试完成"

# 主函数
def main():
    print("=== 鸢尾花分类项目 ===")
    
    # 1. 项目初始化
    project_initialization()
    
    # 2. 数据收集与探索
    df, feature_names, target_names = data_collection_exploration()
    
    # 3. 数据预处理
    X_train, X_test, y_train, y_test, scaler = data_preprocessing(df)
    
    # 4. 模型训练与调优
    best_model, accuracy = model_training(X_train, y_train, X_test, y_test)
    
    # 5. API开发
    app = api_development()
    
    # 6. Docker配置
    docker_configuration()
    
    # 7. 项目文档
    project_documentation()
    
    # 8. 项目测试
    project_testing()
    
    print("\n=== 项目完成 ===")
    print(f"模型准确率: {accuracy:.4f}")
    print("项目已成功完成，包含数据探索、模型训练、API开发和Docker配置")

if __name__ == "__main__":
    main()
