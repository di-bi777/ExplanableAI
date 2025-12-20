import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import LabelEncoder
from graphviz import Digraph
from collections import Counter


class TreeNode:
    """決定木の各ノード"""

    def __init__(self, left=None, right=None, condition=None, labels=None, miss=None):
        self.left = left
        self.right = right
        self.condition = (0, 0)  # (i,threshold) x_i <= threshold  or  x_i > threshold
        self.labels = labels
        self.miss = miss


def minimum_center(i, labels, centers):
    """指定された特徴量iに対するクラスタ中心の最小値を計算"""
    minimum = np.inf
    for j in labels:
        minimum = min(minimum, centers[j][i])
    return minimum


def maximum_center(i, labels, centers):
    """指定された特徴量iに対するクラスタ中心の最大値を計算"""
    maximum = -np.inf
    for j in labels:
        maximum = max(maximum, centers[j][i])
    return maximum


def mistake(x, center, i, threshold):
    """データポイントがクラスタ中心と同じ側にあるかどうかを判定し、異なる場合は1を返す"""
    return 0 if ((x[i] <= threshold) == (center[i] <= threshold)) else 1


def delete_mistakes_data(X, labels, centers, i, threshold):
    """誤分類されたデータポイントを削除し、新しいデータセットとラベルを返す"""
    new_data = []
    new_labels = []
    for idx, x in enumerate(X):
        if mistake(x, centers[labels[idx]], i, threshold) == 0:
            new_data.append(x)
            new_labels.append(labels[idx])
    return np.array(new_data), np.array(new_labels)


def make_next_data(X, labels, i, threshold):
    """データを指定された閾値で左右の子ノードに分割"""
    l_data = []
    l_labels = []
    r_data = []
    r_labels = []
    for idx, x in enumerate(X):
        if x[i] <= threshold:
            l_data.append(x)
            l_labels.append(labels[idx])
        else:
            r_data.append(x)
            r_labels.append(labels[idx])

    return np.array(l_data), np.array(l_labels), np.array(r_data), np.array(r_labels)


def count_mistakes(X, l, i, labels, centers):
    """指定された特徴量と閾値に対する誤分類の数をカウント"""
    cnt = 0
    for idx, x in enumerate(X):
        if mistake(x, centers[labels[idx]], i, l[i]) == 1:
            cnt += 1
    return cnt


def get_best_splits(X, l, r, labels, centers):
    """各特徴量に対する誤分類の数を計算し、最小の誤分類数を持つ分割点を返す"""
    bests_split = {"mistake": np.inf, "coordinate": None, "threshold": None}
    data_dimentions = X.shape[1]

    for i in range(data_dimentions):
        ith_sorted_X = X[X[:, i].argsort(), :]
        ith_sorted_centers = centers[centers[:, i].argsort(), :]
        idx_center = 1
        cnt_mistakes = count_mistakes(X, l, i, labels, centers)
        for j, x in enumerate(ith_sorted_X[:-1]):
            if l[i] > x[i] or x[i] >= r[i]:
                continue

            cnt_mistakes = count_mistakes(X, x, i, labels, centers)

            if bests_split["mistake"] > cnt_mistakes:
                bests_split["mistake"] = cnt_mistakes
                bests_split["coordinate"] = i
                bests_split["threshold"] = x[i]

    return bests_split["coordinate"], bests_split["threshold"], bests_split["mistake"]


def build_tree(X, labels, centers, df):
    """再帰的に決定木を構築"""
    node = TreeNode()
    l = []
    r = []

    if len(np.unique(labels)) == 1:  # クラスターが一つの時
        node.labels = labels
        return node

    for i in range(X.shape[1]):  # 各特徴量についてループ
        l.append(minimum_center(i, labels, centers))
        r.append(maximum_center(i, labels, centers))

    i, threshold, miss = get_best_splits(X, l, r, labels, centers)
    X, labels = delete_mistakes_data(X, labels, centers, i, threshold)
    left_data, left_labels, right_data, right_labels = make_next_data(
        X, labels, i, threshold
    )

    # 特徴量を文字列に変換
    column_names = df.columns.tolist()
    i = column_names[i]

    node.condition = (i, threshold)
    node.miss = miss
    node.labels = labels
    node.left = build_tree(left_data, left_labels, centers, df)
    node.right = build_tree(right_data, right_labels, centers, df)

    return node


def assign_leaf_to_cluster(node):
    counter = Counter(node.labels)
    most_common_label = counter.most_common(1)[0][0]
    return most_common_label


def visualize_tree(node, G=None, id=0):
    """木を描写"""
    if G is None:
        G = Digraph(format="png")
        G.attr("node", shape="circle")

    if node.left or node.right:
        G.node(
            str(id),
            "{} <= {}\nnum of mistakes at this node =>{}".format(
                node.condition[0], node.condition[1], node.miss
            ),
        )
    else:
        most_common_label = assign_leaf_to_cluster(node)
        G.node(str(id), str(most_common_label))

    if node.left:
        left_id = id * 2 + 1
        G.edge(str(id), str(left_id), label="True")
        visualize_tree(node.left, G, left_id)

    if node.right:
        right_id = id * 2 + 2
        G.edge(str(id), str(right_id), label="False")
        visualize_tree(node.right, G, right_id)

    return G


# タイトル
st.title("Explainable AI App")
st.write("kmeansによるクラスタリングを決定木で近似して説明可能性を向上")
st.markdown("<br>", unsafe_allow_html=True)

csv_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

if csv_file is not None:
    # CSVファイルをデータフレームに読み込む
    df = pd.read_csv(csv_file, header=None)
    st.markdown("<br>", unsafe_allow_html=True)

    # 入力されたファイルを表示
    st.write("あなたが入力したファイル")
    st.write(df)
    st.markdown("<br>", unsafe_allow_html=True)

    # 使用する特徴量の列を選択
    feature_column = st.multiselect(
        "クラスタリングに用いる特徴量の列を選択してください", df.columns
    )

    if feature_column:
        # 選択された特徴量のデータフレームを作成
        selected_df = df[feature_column]

        st.write(selected_df)

        # 数値に変換する必要がある列を確認
        label_encoders = {}
        for column in selected_df.columns:
            if (
                isinstance(selected_df[column].iloc[1], str)
                and not selected_df[column].iloc[1].isdigit()
            ):
                le = LabelEncoder()
                selected_df[column].iloc[1:] = le.fit_transform(
                    selected_df[column].iloc[1:]
                )
                label_encoders[column] = le
            elif (
                isinstance(selected_df[column].iloc[1], str)
                and selected_df[column].iloc[1].isdigit()
            ):
                selected_df[column].iloc[1:] = (
                    selected_df[column].iloc[1:].apply(pd.to_numeric)
                )

        # 数値変換の対応表を表示
        for column, le in label_encoders.items():
            st.write(f"'{column}'の数値変換対応表:")
            st.write(dict(zip(le.classes_, le.transform(le.classes_))))

        st.write(selected_df)

        # クラスタ数の入力
        n_clusters = st.number_input(
            "クラスタ数を入力してください", min_value=2, max_value=10, value=3, step=1
        )
        st.markdown("<br>", unsafe_allow_html=True)
        button1 = st.button("クラスタリング開始")

        if button1:
            # KMeansクラスタリングの実行
            selected_df.columns = selected_df.iloc[0]
            selected_df = selected_df[1:]
            selected_df = selected_df.reset_index(drop=True)
            data_array = selected_df.to_numpy()
            kmeans_model = KMeans(n_clusters=n_clusters, random_state=0).fit(data_array)
            labels = kmeans_model.labels_
            centers = kmeans_model.cluster_centers_

            # クラスタリング結果をデータフレームに追加
            clustered_df = selected_df.copy()
            clustered_df["Cluster"] = labels

            # クラスタリング結果を表示
            st.write("クラスタリング結果")
            st.write(clustered_df)

            st.write("クラスタリングの説明（決定木による近似結果）")

            root = build_tree(data_array, labels, centers, selected_df)
            G = visualize_tree(root)
            st.graphviz_chart(G.source)
