#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第28天：树
数据结构与算法学习示例
内容：树的基本概念、二叉树、二叉搜索树
"""

print("=== 第28天：树 ===")

# 1. 树的基本概念
print("\n1. 树的基本概念")

print("树是一种非线性数据结构，由节点和边组成")
print("- 根节点：树的最顶层节点")
print("- 叶子节点：没有子节点的节点")
print("- 父节点：有子节点的节点")
print("- 子节点：被父节点直接连接的节点")
print("- 兄弟节点：具有相同父节点的节点")
print("- 深度：从根节点到该节点的路径长度")
print("- 高度：从该节点到最远叶子节点的路径长度")

# 2. 二叉树
print("\n2. 二叉树")

print("二叉树是每个节点最多有两个子节点的树")

class TreeNode:
    def __init__(self, value):
        """初始化树节点"""
        self.value = value
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self, root=None):
        """初始化二叉树"""
        self.root = root
    
    def insert(self, value):
        """插入节点"""
        if not self.root:
            self.root = TreeNode(value)
            return
        self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        """递归插入节点"""
        if value < node.value:
            if not node.left:
                node.left = TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if not node.right:
                node.right = TreeNode(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        """搜索节点"""
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        """递归搜索节点"""
        if not node:
            return False
        if node.value == value:
            return True
        if value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)
    
    def inorder_traversal(self):
        """中序遍历"""
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        """递归中序遍历"""
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)
    
    def preorder_traversal(self):
        """前序遍历"""
        result = []
        self._preorder_recursive(self.root, result)
        return result
    
    def _preorder_recursive(self, node, result):
        """递归前序遍历"""
        if node:
            result.append(node.value)
            self._preorder_recursive(node.left, result)
            self._preorder_recursive(node.right, result)
    
    def postorder_traversal(self):
        """后序遍历"""
        result = []
        self._postorder_recursive(self.root, result)
        return result
    
    def _postorder_recursive(self, node, result):
        """递归后序遍历"""
        if node:
            self._postorder_recursive(node.left, result)
            self._postorder_recursive(node.right, result)
            result.append(node.value)
    
    def level_order_traversal(self):
        """层序遍历"""
        if not self.root:
            return []
        result = []
        queue = [self.root]
        while queue:
            level_size = len(queue)
            level = []
            for _ in range(level_size):
                node = queue.pop(0)
                level.append(node.value)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(level)
        return result
    
    def get_height(self):
        """获取树的高度"""
        return self._height_recursive(self.root)
    
    def _height_recursive(self, node):
        """递归获取树的高度"""
        if not node:
            return 0
        left_height = self._height_recursive(node.left)
        right_height = self._height_recursive(node.right)
        return max(left_height, right_height) + 1
    
    def is_balanced(self):
        """检查树是否平衡"""
        return self._is_balanced_recursive(self.root)[0]
    
    def _is_balanced_recursive(self, node):
        """递归检查树是否平衡"""
        if not node:
            return True, 0
        left_balanced, left_height = self._is_balanced_recursive(node.left)
        right_balanced, right_height = self._is_balanced_recursive(node.right)
        balanced = left_balanced and right_balanced and abs(left_height - right_height) <= 1
        height = max(left_height, right_height) + 1
        return balanced, height

# 测试二叉树
print("\n测试二叉树操作:")
tree = BinaryTree()
tree.insert(50)
tree.insert(30)
tree.insert(70)
tree.insert(20)
tree.insert(40)
tree.insert(60)
tree.insert(80)

print(f"中序遍历: {tree.inorder_traversal()}")
print(f"前序遍历: {tree.preorder_traversal()}")
print(f"后序遍历: {tree.postorder_traversal()}")
print(f"层序遍历: {tree.level_order_traversal()}")
print(f"树的高度: {tree.get_height()}")
print(f"树是否平衡: {tree.is_balanced()}")
print(f"搜索节点40: {tree.search(40)}")
print(f"搜索节点90: {tree.search(90)}")

# 3. 二叉搜索树
print("\n3. 二叉搜索树")

print("二叉搜索树是一种特殊的二叉树，左子节点的值小于父节点，右子节点的值大于父节点")

class BinarySearchTree(BinaryTree):
    def delete(self, value):
        """删除节点"""
        self.root = self._delete_recursive(self.root, value)
    
    def _delete_recursive(self, node, value):
        """递归删除节点"""
        if not node:
            return node
        if value < node.value:
            node.left = self._delete_recursive(node.left, value)
        elif value > node.value:
            node.right = self._delete_recursive(node.right, value)
        else:
            # 叶子节点或只有一个子节点的情况
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            # 有两个子节点的情况，找到右子树的最小值
            min_node = self._find_min(node.right)
            node.value = min_node.value
            node.right = self._delete_recursive(node.right, min_node.value)
        return node
    
    def _find_min(self, node):
        """找到最小节点"""
        current = node
        while current.left:
            current = current.left
        return current
    
    def find_min(self):
        """查找树中的最小值"""
        if not self.root:
            return None
        return self._find_min(self.root).value
    
    def find_max(self):
        """查找树中的最大值"""
        if not self.root:
            return None
        current = self.root
        while current.right:
            current = current.right
        return current.value

# 测试二叉搜索树
print("\n测试二叉搜索树操作:")
bst = BinarySearchTree()
bst.insert(50)
bst.insert(30)
bst.insert(70)
bst.insert(20)
bst.insert(40)
bst.insert(60)
bst.insert(80)

print(f"中序遍历: {bst.inorder_traversal()}")
print(f"树中的最小值: {bst.find_min()}")
print(f"树中的最大值: {bst.find_max()}")

print("删除节点20:")
bst.delete(20)
print(f"中序遍历: {bst.inorder_traversal()}")

print("删除节点30:")
bst.delete(30)
print(f"中序遍历: {bst.inorder_traversal()}")

print("删除节点50:")
bst.delete(50)
print(f"中序遍历: {bst.inorder_traversal()}")

# 4. 平衡二叉树
print("\n4. 平衡二叉树")

print("平衡二叉树是一种特殊的二叉搜索树，左右子树的高度差不超过1")

class AVLTreeNode(TreeNode):
    def __init__(self, value):
        super().__init__(value)
        self.height = 1

class AVLTree:
    def __init__(self):
        """初始化AVL树"""
        self.root = None
    
    def get_height(self, node):
        """获取节点的高度"""
        if not node:
            return 0
        return node.height
    
    def get_balance(self, node):
        """获取节点的平衡因子"""
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)
    
    def right_rotate(self, z):
        """右旋操作"""
        y = z.left
        T3 = y.right
        
        y.right = z
        z.left = T3
        
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        
        return y
    
    def left_rotate(self, z):
        """左旋操作"""
        y = z.right
        T2 = y.left
        
        y.left = z
        z.right = T2
        
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        
        return y
    
    def insert(self, value):
        """插入节点"""
        self.root = self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        """递归插入节点"""
        if not node:
            return AVLTreeNode(value)
        if value < node.value:
            node.left = self._insert_recursive(node.left, value)
        elif value > node.value:
            node.right = self._insert_recursive(node.right, value)
        else:
            return node
        
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))
        
        balance = self.get_balance(node)
        
        # 左左情况
        if balance > 1 and value < node.left.value:
            return self.right_rotate(node)
        # 右右情况
        if balance < -1 and value > node.right.value:
            return self.left_rotate(node)
        # 左右情况
        if balance > 1 and value > node.left.value:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)
        # 右左情况
        if balance < -1 and value < node.right.value:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)
        
        return node
    
    def inorder_traversal(self):
        """中序遍历"""
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        """递归中序遍历"""
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)
    
    def get_root_height(self):
        """获取根节点的高度"""
        if not self.root:
            return 0
        return self.root.height

# 测试AVL树
print("\n测试AVL树操作:")
avl = AVLTree()
avl.insert(10)
avl.insert(20)
avl.insert(30)
avl.insert(40)
avl.insert(50)
avl.insert(25)

print(f"中序遍历: {avl.inorder_traversal()}")
print(f"AVL树的高度: {avl.get_root_height()}")

# 5. 时间复杂度分析
print("\n5. 时间复杂度分析")

print("二叉搜索树的时间复杂度:")
print("- 插入: O(h)，其中h是树的高度")
print("- 删除: O(h)")
print("- 搜索: O(h)")

print("\n平衡二叉树的时间复杂度:")
print("- 插入: O(log n)")
print("- 删除: O(log n)")
print("- 搜索: O(log n)")

# 6. 应用场景
print("\n6. 应用场景")

print("树的应用场景:")
print("- 二叉搜索树: 字典、集合、数据库索引")
print("- AVL树: 需要平衡的搜索场景")
print("- 红黑树: 平衡树的另一种实现，常用于C++的STL")
print("- B树: 数据库和文件系统的索引结构")
print("- 堆: 优先队列、堆排序")
print("- Trie树: 字符串搜索、自动补全")

# 7. 练习
print("\n7. 练习")

# 练习1: 验证二叉搜索树
print("练习1: 验证二叉搜索树")
print("- 实现一个函数来验证一棵二叉树是否是有效的二叉搜索树")
print("- 测试不同的二叉树")

# 练习2: 二叉树的最大深度
print("\n练习2: 二叉树的最大深度")
print("- 实现一个函数来计算二叉树的最大深度")
print("- 测试不同的二叉树")

# 练习3: 二叉树的最小深度
print("\n练习3: 二叉树的最小深度")
print("- 实现一个函数来计算二叉树的最小深度")
print("- 测试不同的二叉树")

# 练习4: 二叉树的镜像
print("\n练习4: 二叉树的镜像")
print("- 实现一个函数来获取二叉树的镜像")
print("- 测试镜像操作")

# 练习5: 二叉树的路径和
print("\n练习5: 二叉树的路径和")
print("- 实现一个函数来检查二叉树中是否存在从根到叶子的路径和等于目标值")
print("- 测试不同的目标值")

print("\n=== 第28天学习示例结束 ===")
