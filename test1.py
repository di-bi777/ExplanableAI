import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph
from sklearn.tree import export_graphviz,DecisionTreeClassifier
from collections import Counter

class TreeNode:
    """決定木の各ノード"""
    def __init__(self, left=None, right=None, condition=None, labels = None, miss = None):
        self.left = left
        self.right = right 
        self.condition = (0,0) #(i,threshold) x_i <= threshold  or  x_i > threshold 
        self.labels = labels
        self.miss = miss

def minimum_center(i,labels,centers):
    """指定された特徴量iに対するクラスタ中心の最小値を計算"""
    minimum = np.inf
    for j in labels:
        minimum = min(minimum, centers[j][i])
    return minimum

def maximum_center(i,labels,centers):
    """指定された特徴量iに対するクラスタ中心の最大値を計算"""
    maximum = -np.inf
    for j in labels:
        maximum = max(maximum, centers[j][i])
    return maximum  

def mistake(x,center,i,threshold):
    """データポイントがクラスタ中心と同じ側にあるかどうかを判定し、異なる場合は1を返す"""
    return 0 if ((x[i]<=threshold) == (center[i]<=threshold)) else 1

def delete_mistakes_data(X,labels,centers,i,threshold):
    """誤分類されたデータポイントを削除し、新しいデータセットとラベルを返す"""
    new_data = []
    new_labels=[]
    for idx,x in enumerate(X):
        if(mistake(x,centers[labels[idx]],i,threshold)==0):
            new_data.append(x)
            new_labels.append(labels[idx])
    return np.array(new_data),np.array(new_labels)
    
def make_next_data(X,labels,i,threshold):
    """データを指定された閾値で左右の子ノードに分割"""
    l_data=[]
    l_labels=[]
    r_data=[]
    r_labels=[]
    for idx,x in enumerate(X):
        if(x[i]<=threshold):
            l_data.append(x)
            l_labels.append(labels[idx])
        else:
            r_data.append(x)
            r_labels.append(labels[idx])
            
    return np.array(l_data),np.array(l_labels),np.array(r_data),np.array(r_labels)

def count_mistakes(X,l,i,labels,centers):
    """指定された特徴量と閾値に対する誤分類の数をカウント"""
    cnt=0
    for idx,x in enumerate(X):
        if(mistake(x,centers[labels[idx]],i,l[i])==1):
            cnt+=1
    return cnt

def get_best_splits(X,l,r,labels,centers):
    """各特徴量に対する誤分類の数を計算し、最小の誤分類数を持つ分割点を返す"""
    bests_split = {'mistake':np.inf,'coordinate':None,'threshold':None}
    data_dimentions = X.shape[1]
    
    for i in range(data_dimentions):
        ith_sorted_X = X[X[:,i].argsort(), :]
        ith_sorted_centers = centers[centers[:,i].argsort(), :]
        idx_center = 1
        cnt_mistakes = count_mistakes(X,l,i,labels,centers)
        for j,x in enumerate(ith_sorted_X[:-1]):
            if(l[i]>x[i] or x[i]>=r[i]):
                continue
                
            cnt_mistakes = count_mistakes(X,x,i,labels,centers) #ここで本来はDPでより効率よく計算すべきだが，やり方がよくわからない．なのでナイーブなやり方でやっている．つまり，全データに対してその分割でmistakeとなるのか否かを調べている
            
            if bests_split['mistake'] > cnt_mistakes:
                bests_split['mistake'] = cnt_mistakes
                bests_split['coordinate'] = i
                bests_split['threshold'] = x[i]   
    print("num of mistakes at this node => {}".format(bests_split['mistake']))
    return bests_split['coordinate'],bests_split['threshold'],bests_split['mistake']

def build_tree(X,labels,centers,df):
    """再帰的に決定木を構築"""
    node = TreeNode()
    l=[]
    r=[]
    
    if(len(np.unique(labels))==1):#クラスターが一つの時
        node.labels = labels
        return node

    for i in range(X.shape[1]):#各特徴量についてループ
        l.append(minimum_center(i,labels,centers))
        r.append(maximum_center(i,labels,centers))

    i,threshold,miss = get_best_splits(X,l,r,labels,centers)
    X,labels = delete_mistakes_data(X,labels,centers,i,threshold)
    left_data,left_labels,right_data,right_labels = make_next_data(X,labels,i,threshold)
    
    #特徴量を文字列に変換
    column_names = df.columns.tolist()
    i = column_names[i]
    
    node.condition = (i,threshold)
    node.labels = labels
    node.miss = miss
    node.left = build_tree(left_data,left_labels,centers,df)
    node.right = build_tree(right_data,right_labels,centers,df)
    
    return node

def assign_leaf_to_cluster(node):
    counter = Counter(node.labels)
    most_common_label = counter.most_common(1)[0][0]
    return most_common_label

def visualize_tree(node,G=None, id=0):
    """木を描写"""
    if G is None:
        G = Digraph(format='png')
        G.attr('node', shape='circle')
        
    if node.left or node.right:
        G.node(str(id), "{} <= {}\nnum of mistakes at this node =>{}".format(node.condition[0],node.condition[1],node.miss))
    else:
        most_common_label = assign_leaf_to_cluster(node)
        G.node(str(id), str(most_common_label))
        
    if node.left:
        left_id = id * 2 + 1
        G.edge(str(id), str(left_id),label="True")
        visualize_tree(node.left, G, left_id)

    if node.right:
        right_id = id * 2 + 2
        G.edge(str(id), str(right_id),label="False")
        visualize_tree(node.right, G, right_id)

    return G

def display_tree(G):
    """グラフを表示"""
    G.render('tree', view=True)

# Kaggleの「Mall Customers」データセットを読み込む
file_path = 'C:\\Users\\FujimotoD\\OneDrive\\documents\\intern\\ExplanableAI\\Mall_Customers.csv'
data = pd.read_csv(file_path)

#IDの列を消す
del(data['CustomerID'])

# 'Genre'列を数値に変換
data['Genre'] = data['Genre'].map({'Male': 0, 'Female': 1})

#NumPy配列へ変換
data_array = np.array([data['Genre'].tolist(),
                       data['Age'].tolist(),
                       data['Annual Income (k$)'].tolist(),
                       data['Spending Score (1-100)'].tolist()
                       ], np.int32)

data_array = data_array.T

# KMeansクラスタリングを適用する
kmeans_model = KMeans(n_clusters=5, random_state=0).fit(data_array)

centers = kmeans_model.cluster_centers_
labels = kmeans_model.labels_

root = build_tree(data_array,labels,centers,data)
G = visualize_tree(root)
display_tree(G)

"""
# クラスタリング結果をデータフレームに追加する
data['Cluster'] = clusters
"""
"""
# クラスタリング結果を近似するための決定木の訓練
tree = DecisionTreeClassifier(max_leaf_nodes=4, random_state=0)
tree.fit(X, data['Cluster'])

# 決定木のプロット
plt.figure(figsize=(20,10))
plot_tree(tree, feature_names=['Annual Income (k$)', 'Spending Score (1-100)'], class_names=[str(i) for i in range(5)], filled=True)
plt.show()

# 決定木のルールを表示
from sklearn.tree import export_text
tree_rules = export_text(tree, feature_names=['Annual Income (k$)', 'Spending Score (1-100)'])
print(tree_rules)
"""

"""
# クラスタリング結果を色分けして表示する
plt.figure(figsize=(10, 6))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Mall Customers Clustering')
plt.colorbar(label='Cluster')
plt.show()
"""