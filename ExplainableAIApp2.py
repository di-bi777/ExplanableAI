import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from graphviz import Digraph
from collections import Counter
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

os.environ["OMP_NUM_THREADS"] = "1"


class TreeNode:
    """決定木の各ノード"""

    def __init__(
        self, left=None, right=None, condition=None, labels=None, miss=None, data=None
    ):
        self.left = left  # 閾値以下
        self.right = right
        self.condition = (0, 0)  # (i,threshold) x_i <= threshold  or  x_i > threshold
        self.labels = labels  # kmeans.labels_ で取得
        self.miss = miss
        self.data = data


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
    return (
        0 if ((x[i] <= threshold) == (center[i] <= threshold)) else 1
    )  # x[i]は各データ点のi番目の特徴量の数


def delete_mistakes_data(X, labels, centers, i, threshold):
    """ミスとなったデータポイントを削除し、新しいデータセットとラベルを返す"""
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
    """指定された特徴量と閾値に対するミスの数をカウント"""
    cnt = 0
    for idx, x in enumerate(X):
        if mistake(x, centers[labels[idx]], i, l[i]) == 1:
            cnt += 1
    return cnt


def get_best_splits(X, l, r, labels, centers):
    """各特徴量に対するミスの数を計算し、最小のミス数を持つ分割点を返す"""
    bests_split = {"mistake": np.inf, "coordinate": None, "threshold": None}
    data_dimentions = X.shape[1]

    for i in range(data_dimentions):
        ith_sorted_X = X[X[:, i].argsort(), :]
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
        node.data = X
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
    node.data = X
    node.left = build_tree(left_data, left_labels, centers, df)
    node.right = build_tree(right_data, right_labels, centers, df)

    return node


def assign_leaf_to_cluster(node):
    counter = Counter(node.labels)
    most_common_label = counter.most_common(1)[0][0]
    return most_common_label


def create_histogram(node, i, threshold, id, columns):
    """各ノードのヒストグラムを作る"""

    # numpy配列をデータフレームに変換
    new_column_array = np.array(node.labels).reshape(-1, 1)
    updated_array = np.column_stack((node.data, new_column_array))
    df = pd.DataFrame(updated_array, columns=columns)

    # プロットの設定
    fig, ax = plt.subplots(figsize=(4, 3))

    # クラスターごとの一貫した色を定義
    colors = [
        "#FEFEBB",
        "#c7e9b4",
        "#41b6c4",
        "#74add1",
        "#4575b4",
        "#313695",
        "#fee090",
        "#fdae61",
        "#f46d43",
        "#d73027",
    ]
    cluster_colors = {
        cluster: colors[np.int_(cluster)] for cluster in sorted(df["Cluster"].unique())
    }

    # ビンの範囲を設定
    bins = np.linspace(df[i].min(), df[i].max(), 21)

    # クラスターごとにヒストグラムを描画
    n, bins, patches = ax.hist(
        [df[df["Cluster"] == cluster][i] for cluster in sorted(df["Cluster"].unique())],
        bins=bins,
        stacked=True,
        color=[cluster_colors[cluster] for cluster in sorted(df["Cluster"].unique())],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        label=[f"Cluster {cluster}" for cluster in sorted(df["Cluster"].unique())],
    )

    # 閾値を示す垂直線
    ax.axvline(x=threshold, color="red", linestyle="--", linewidth=1, label="Threshold")

    # グラフの装飾
    ax.set_xlabel(i)
    ax.set_ylabel("Frequency")
    ax.set_title(f"{i} <= {threshold}\nnum of mistake = {node.miss}")

    # 凡例
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # グリッド線を追加
    ax.grid(True, linestyle="--", alpha=0.7)

    # グラフを保存
    plt.tight_layout()
    filename = f"node_{id}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    return filename


def visualize_tree(node, columns, G=None, id=0, i=0):
    """木を描写"""

    if G is None:
        G = Digraph(format="png")
        G.attr(
            "node",
            shape="circle",
            fontname="Helvetica",
            fixedsize="true",
            width="1.5",
            height="1.5",
        )
        G.attr("edge", color="gray", fontname="Helvetica")

    current_i = i

    if node.left or node.right:
        current_i += 1
        hist_file = create_histogram(
            node, node.condition[0], node.condition[1], id, columns
        )
        G.node(str(id), f"Split{current_i}")
        st.write(
            f"Split{current_i}: {node.condition[0]}<={node.condition[1]}, num of mistake at this node = {node.miss}"
        )
        expander = st.expander(f"More information of Split{current_i}")
        with expander:
            st.image(hist_file, use_container_width=True)

    else:
        most_common_label = assign_leaf_to_cluster(node)
        G.node(str(id), str(most_common_label))

    if node.left:
        left_id = id * 2 + 1
        G.edge(str(id), str(left_id), label="True")
        current_i, G = visualize_tree(node.left, columns, G, left_id, current_i)

    if node.right:
        right_id = id * 2 + 2
        G.edge(str(id), str(right_id), label="False")
        current_i, G = visualize_tree(node.right, columns, G, right_id, current_i)

    return current_i, G


# タイトル
st.title("Explainable AI App")
st.write("クラスタリングを決定木で近似して説明可能性を向上")
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

    # クラスタ数の入力
    n_clusters = st.number_input(
        "クラスタリングを行う際のクラスタ数を入力してください",
        min_value=2,
        max_value=10,
        value=3,
        step=1,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("PCA及びUMAPで次元削減を行います")
    # 元のデータの次元数を表示
    original_dimensions = df.shape[1]
    st.write(f"元のデータの次元数: {original_dimensions}")
    # PCAの次元数を入力
    n_pca_components = st.number_input(
        "PCAで何次元まで圧縮するかを入力してください",
        min_value=2,
        max_value=original_dimensions,
        value=original_dimensions,
    )
    st.write(f"PCAによる次元圧縮: {original_dimensions}->{n_pca_components}")
    # 次元数の選択
    n_umap_components = st.radio("UMAPで何次元まで圧縮するか選択してください:", [2, 3])
    st.write(f"UMAPによる次元圧縮: {n_pca_components}->{n_umap_components}")
    button1 = st.button("次元圧縮・クラスタリング開始")

    if button1:
        # KMeansクラスタリングの実行
        df.columns = df.iloc[0]
        df = df[1:]
        df = df.reset_index(drop=True)
        data_array = df.to_numpy()
        # 文字列を浮動小数点数に変換
        data_array = data_array.astype(float)
        # 標準化
        sc = StandardScaler()
        X_std = sc.fit_transform(data_array)

        # PCAによる次元圧縮
        pca = PCA(n_components=n_pca_components, random_state=42)
        X_pca = pca.fit_transform(X_std)

        # UMAPによる次元削減
        reducer = umap.UMAP(n_components=n_umap_components)
        embedding = reducer.fit_transform(X_pca)

        # KMeansクラスタリング
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embedding)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # クラスタ中心をPCA前の次元に戻す
        # UMAP -> PCA -> 標準化前の次元に逆変換する
        centers_pca = reducer.inverse_transform(centers)
        centers_original = pca.inverse_transform(centers_pca)
        centers_original_unscaled = sc.inverse_transform(centers_original)

        # プロットの作成
        fig = plt.figure()
        if n_umap_components == 3:
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                embedding[:, 2],
                c=labels,
                cmap="viridis",
                s=50,
                label="Data Points",
            )
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                centers[:, 2],
                c="red",
                s=200,
                alpha=0.75,
                marker="X",
                label="Cluster Centers",
            )
            ax.set_title(
                "KMeans Clustering of Wine Data after UMAP Dimensionality Reduction"
            )
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.set_zlabel("UMAP Dimension 3")
            cbar = plt.colorbar(sc, ax=ax, label="Cluster Label")
            ax.legend()
        else:
            plt.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels,
                cmap="viridis",
                s=50,
                label="Data Points",
            )
            plt.scatter(
                centers[:, 0],
                centers[:, 1],
                c="red",
                s=200,
                alpha=0.75,
                marker="X",
                label="Cluster Centers",
            )
            plt.title(
                "KMeans Clustering of Wine Data after UMAP Dimensionality Reduction"
            )
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dimension 2")
            plt.legend()

        # クラスタリング結果を表示
        st.write("クラスタリング結果")
        st.pyplot(fig)
        # クラスタリング結果をデータフレームに追加
        clustered_df = df.copy()
        clustered_df["Cluster"] = labels

        # 決定木による近似結果を表示
        st.write("クラスタリングの説明（決定木による近似結果）")
        root = build_tree(data_array, labels, centers_original_unscaled, df)
        total_split, G = visualize_tree(root, clustered_df.columns)
        st.graphviz_chart(G.source)
