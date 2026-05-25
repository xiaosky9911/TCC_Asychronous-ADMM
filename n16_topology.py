# -*- coding: utf-8 -*-
"""
生成16节点网络拓扑图：
- 读取 need_data_N16_improved.xlsx 或 need_data_N16.xlsx 中的 '带宽上限' 表
- 根据带宽绘制带权无向图，节点 0-7 为本地集群，8-15 为远程集群
- 输出文件：n16_topology.png
"""
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

# 可调整的参数
INPUT_FILES = ['need_data_N16_improved.xlsx', 'need_data_N16.xlsx']
OUT_FILE = 'n16_topology.png'
EDGE_THRESHOLD = 0.3  # 只绘制权重大于该阈值的边

# 找到文件
base_dir = os.path.abspath(os.path.dirname(__file__))
for f in INPUT_FILES:
    path = os.path.join(base_dir, f)
    if os.path.exists(path):
        bw_file = path
        break
else:
    raise FileNotFoundError('未找到 need_data_N16_improved.xlsx 或 need_data_N16.xlsx')

# 读取带宽矩阵
df_bw = pd.read_excel(bw_file, sheet_name='带宽上限', index_col=0)
# 有些表格第一列是索引列，取所有数值列
bw_mat = df_bw.values.astype(float)
# 如果表里包含对角索引列（比如第一列是索引），确保矩阵是方阵
n = bw_mat.shape[0]
if n != 16:
    # 若文件包含首列作为说明列，尝试取第1:17列
    if df_bw.shape[1] >= 17:
        bw_mat = df_bw.iloc[:,1:17].values.astype(float)
        n = bw_mat.shape[0]

if bw_mat.shape[0] != 16 or bw_mat.shape[1] != 16:
    raise ValueError(f'读取到的带宽矩阵不是16x16，shape={bw_mat.shape}')

# 将对角置为0以免影响绘图
np.fill_diagonal(bw_mat, 0.0)

# 创建图
G = nx.Graph()
G.add_nodes_from(range(16))

# 添加边（仅添加权重大于阈值的边）
for i in range(16):
    for j in range(i+1, 16):
        w = float(bw_mat[i, j])
        if w > EDGE_THRESHOLD:
            G.add_edge(i, j, weight=w)

# 布局：将本地集群和远程集群分开布局
pos = {}
# 左侧圆圈：0-7
theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
r_left = 1.0
for idx, node in enumerate(range(0,8)):
    pos[node] = (-1.5 + r_left*np.cos(theta[idx]), r_left*np.sin(theta[idx]))
# 右侧圆圈：8-15
theta2 = np.linspace(0, 2*np.pi, 8, endpoint=False)
r_right = 1.0
for idx, node in enumerate(range(8,16)):
    pos[node] = (1.5 + r_right*np.cos(theta2[idx]), r_right*np.sin(theta2[idx]))

# 节点颜色按集群
colors = ['#1f78b4' if i < 8 else '#33a02c' for i in range(16)]

# 画图
plt.figure(figsize=(10,6))
# 画所有节点（标签）
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=600, alpha=0.9)
# 画标签
labels = {i: str(i) for i in range(16)}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='white')

# 画边，按权重缩放宽度与透明度
edges = G.edges(data=True)
if len(edges) > 0:
    weights = np.array([d['weight'] for (_,_,d) in edges])
    # 线宽缩放
    w_min, w_max = weights.min(), weights.max()
    if w_min == w_max:
        widths = [2.0 for _ in weights]
    else:
        widths = 1.0 + 4.0 * (weights - w_min) / (w_max - w_min)
    # 透明度
    alphas = 0.3 + 0.7 * (weights - w_min) / (w_max - w_min) if w_min != w_max else [0.8 for _ in weights]

    # 分别绘制每条边以支持不同alpha
    for (u,v,d), width, alpha in zip(edges, widths, alphas):
        nx.draw_networkx_edges(G, pos, edgelist=[(u,v)], width=width, edge_color='gray', alpha=alpha)
else:
    print('警告：没有边被绘制（请检查 EDGE_THRESHOLD）')

# 图例自定义
import matplotlib.patches as mpatches
local_patch = mpatches.Patch(color='#1f78b4', label='Local cluster (0-7)')
remote_patch = mpatches.Patch(color='#33a02c', label='Remote cluster (8-15)')
plt.legend(handles=[local_patch, remote_patch], loc='upper center', ncol=2)

plt.axis('off')
plt.title('16-node topology (edge weight = bandwidth)')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, OUT_FILE), dpi=200)
print(f'已生成: {OUT_FILE} (节点数={G.number_of_nodes()}, 边数={G.number_of_edges()})')
