import numpy as np

# 1. 矩阵运算示例
def matrix_operations():
    print("=== 矩阵运算示例 ===")
    
    # 创建矩阵
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    
    print("矩阵 A:")
    print(a)
    print("矩阵 B:")
    print(b)
    
    # 矩阵加法
    c = a + b
    print("\n矩阵加法 A + B:")
    print(c)
    
    # 矩阵乘法
    d = np.dot(a, b)
    print("\n矩阵乘法 A · B:")
    print(d)
    
    # 标量乘法
    e = 2 * a
    print("\n标量乘法 2 · A:")
    print(e)
    
    # 矩阵转置
    f = a.T
    print("\n矩阵转置 A^T:")
    print(f)
    
    # 矩阵求逆
    g = np.linalg.inv(a)
    print("\n矩阵求逆 A^{-1}:")
    print(g)
    
    # 验证逆矩阵
    h = np.dot(a, g)
    print("\n验证 A · A^{-1}:")
    print(h)

# 2. 线性方程组求解示例
def linear_equations():
    print("\n=== 线性方程组求解示例 ===")
    
    # 定义系数矩阵和常数项向量
    A = np.array([[1, 2], [3, 4]])
    b = np.array([5, 11])
    
    print("系数矩阵 A:")
    print(A)
    print("常数项向量 b:")
    print(b)
    
    # 求解线性方程组
    x = np.linalg.solve(A, b)
    print("\n方程组的解 x:")
    print(x)
    
    # 验证解
    print("\n验证 A · x = b:")
    print(np.dot(A, x))

# 3. 特征值与特征向量计算示例
def eigenvalues_eigenvectors():
    print("\n=== 特征值与特征向量计算示例 ===")
    
    # 定义矩阵
    A = np.array([[2, 1], [1, 2]])
    print("矩阵 A:")
    print(A)
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print("\n特征值:")
    print(eigenvalues)
    print("\n特征向量:")
    print(eigenvectors)
    
    # 验证特征值和特征向量
    print("\n验证 A · v = λ · v:")
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lambda_v = eigenvalues[i] * v
        A_v = np.dot(A, v)
        print(f"\n特征值 λ = {eigenvalues[i]}:")
        print(f"A · v = {A_v}")
        print(f"λ · v = {lambda_v}")

# 4. 奇异值分解（SVD）示例
def singular_value_decomposition():
    print("\n=== 奇异值分解（SVD）示例 ===")
    
    # 定义矩阵
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("矩阵 A:")
    print(A)
    
    # 进行SVD分解
    U, S, Vt = np.linalg.svd(A)
    print("\n矩阵 U:")
    print(U)
    print("\n奇异值 S:")
    print(S)
    print("\n矩阵 V^T:")
    print(Vt)
    
    # 重构矩阵
    S_matrix = np.zeros((3, 3))
    S_matrix[:3, :3] = np.diag(S)
    A_reconstructed = U.dot(S_matrix).dot(Vt)
    print("\n重构矩阵 A:")
    print(A_reconstructed)

# 5. PCA算法实现示例
def pca_implementation():
    print("\n=== PCA算法实现示例 ===")
    
    # 生成示例数据
    np.random.seed(42)
    X = np.random.randn(100, 3)  # 100个样本，3个特征
    print("原始数据形状:", X.shape)
    
    # 实现PCA算法
    def pca(X, n_components):
        # 中心化处理
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean
        
        # 计算协方差矩阵
        cov_matrix = np.cov(X_centered.T)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 选择前n_components个最大的特征值对应的特征向量
        indices = np.argsort(eigenvalues)[::-1][:n_components]
        selected_eigenvectors = eigenvectors[:, indices]
        
        # 投影
        X_pca = np.dot(X_centered, selected_eigenvectors)
        
        return X_pca, selected_eigenvectors, X_mean
    
    # 应用PCA降维到2维
    X_pca, eigenvectors, mean = pca(X, 2)
    print("降维后数据形状:", X_pca.shape)
    print("\n主成分（特征向量）:")
    print(eigenvectors)
    print("\n数据均值:")
    print(mean)

# 6. 线性回归示例
def linear_regression():
    print("\n=== 线性回归示例 ===")
    
    # 生成数据
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([[2], [3], [4], [5]])
    
    print("特征矩阵 X:")
    print(X)
    print("目标向量 y:")
    print(y)
    
    # 添加偏置项
    X_b = np.c_[np.ones((4, 1)), X]
    print("\n添加偏置项后的 X_b:")
    print(X_b)
    
    # 计算系数
    beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print("\n回归系数 beta:")
    print(beta)
    
    # 预测
    X_new = np.array([[3, 4]])
    X_new_b = np.c_[np.ones((1, 1)), X_new]
    y_pred = X_new_b.dot(beta)
    print("\n新样本预测值:")
    print(y_pred)

# 运行所有示例
if __name__ == "__main__":
    matrix_operations()
    linear_equations()
    eigenvalues_eigenvectors()
    singular_value_decomposition()
    pca_implementation()
    linear_regression()
