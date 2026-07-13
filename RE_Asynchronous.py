# -*- coding: utf-8 -*-
# 异步边共识 ADMM 算力调度（辅助变量 u 完整版）
# 4节点 / 8节点数据集均可运行
#
# 示例：
# mpiexec -np 4 python d:\yanjiusheng\ready\Cloud_TCC_Revision\RE_Asynchronous.py need_data_N4
# mpiexec -np 8 python d:\yanjiusheng\ready\Cloud_TCC_Revision\RE_Asynchronous.py need_data_N8
# mpiexec -np 16 python d:\yanjiusheng\ready\Cloud_TCC_Revision\RE_Asynchronous.py need_data_N16
# $env:CENTRALIZED_BASE="1182823039.614968"; mpiexec -np 8 python d:\yanjiusheng\ready\Cloud_TCC_Revision\RE_Asynchronous.py need_data_N8

import pandas as pd
from mpi4py import MPI
from timeit import default_timer as timer
import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import copy
import os
import sys
import pickle

# ================= 1. MPI 初始化与全局参数 =================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

base_dir = os.environ.get('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))

DATASET_OPTIONS = {
    "need_data_N4": {
        "file": os.path.join(base_dir, "need_data_N4.xlsx"),
        "I": 4,
        "J": [3, 3, 2, 2],
    },
    "need_data_N8": {
        "file": os.path.join(base_dir, "need_data_N8.xlsx"),
        "I": 8,
        "J": [3, 3, 3, 3, 3, 3, 3, 3],
    },
    "need_data_N16": {
        "file": os.path.join(base_dir, "need_data_N16.xlsx"),
        "I": 16,
        "J": [3] * 16,
    },
    "need_data_N32": {
        "file": os.path.join(base_dir, "need_data_N32.xlsx"),
        "I": 32,
        "J": [3] * 32,
    },
}

# 默认使用 need_data_N4，也可通过命令行参数或环境变量指定
# 命令行参数优先级高于环境变量
# 例如：python RE_Asynchronous_AuxU.py need_data_N8
dataset_name = os.environ.get("DATASET_NAME", "need_data_N4")
if len(sys.argv) > 1:
    dataset_name = sys.argv[1]

# 可选：集中式基准值，用于自动计算相对误差
centralized_base = None
if "CENTRALIZED_BASE" in os.environ:
    centralized_base = float(os.environ["CENTRALIZED_BASE"])

if dataset_name not in DATASET_OPTIONS:
    if rank == 0:
        print(f"未知数据集: {dataset_name}，可选项: {list(DATASET_OPTIONS.keys())}")
    comm.Abort()

cfg = DATASET_OPTIONS[dataset_name]
io = cfg["file"]
I = cfg["I"]
J = cfg["J"]

if size != I:
    if rank == 0:
        print(f"MPI 进程数与数据集节点数不匹配: -np {size}, 数据集要求 {I} (dataset={dataset_name})")
    comm.Abort()

import json as _json

_topo_env = os.environ.get("TOPOLOGY", "full")
if _topo_env == "full":
    neighbors = [j for j in range(I) if j != rank]
else:
    _topo_dict = _json.loads(_topo_env)
    neighbors = sorted(_topo_dict[str(rank)])

# 支持通过环境变量配置掉队者比例与睡眠时长（默认：25% 掉队者，0.04s vs 0.005s）
SLOW_RATIO = float(os.environ.get('SLOW_RATIO', '0.25'))
SLOW_SLEEP = float(os.environ.get('SLOW_SLEEP', '0.04'))
NORMAL_SLEEP = float(os.environ.get('NORMAL_SLEEP', '0.005'))
# 确定掉队者 ranks（取最后若干 ranks，固定）
num_slow = max(1, int(np.ceil(SLOW_RATIO * I)))
slow_ranks = list(range(I - num_slow, I))

# SCENARIO=2/3/5/6 随机时延参数
# SCENARIO=2: 通畅网络——所有节点全程 NORMAL_SLEEP + 高斯抖动
# SCENARIO=3: 固定基础时延 + 对数正态抖动——每个 rank 有不同的固定基础延迟，
#             叠加对数正态随机抖动，模拟节点计算能力异构 + 网络随机抖动
# SCENARIO=5: 固定掉队者基础时延 + 相对随机扰动，便于与 SCENARIO=4 对照
# SCENARIO=6: 固定掉队者身份 + 均匀随机更新时间
CONGESTION_JITTER_STD = float(os.environ.get('CONGESTION_JITTER_STD', '0.003'))  # 高斯抖动标准差（s），仅 SCENARIO=2
LOGNORMAL_MEAN = float(os.environ.get('LOGNORMAL_MEAN', '-4.8'))   # 对数正态均值参数，仅 SCENARIO=3
LOGNORMAL_SIGMA = float(os.environ.get('LOGNORMAL_SIGMA', '0.9'))  # 对数正态标准差参数，仅 SCENARIO=3
RANDOM_DELAY_CV = float(os.environ.get('RANDOM_DELAY_CV', '0.3'))  # 相对随机扰动强度，仅 SCENARIO=5
NORMAL_SLEEP_LOW = float(os.environ.get('NORMAL_SLEEP_LOW', '0.004'))
NORMAL_SLEEP_HIGH = float(os.environ.get('NORMAL_SLEEP_HIGH', '0.006'))
SLOW_SLEEP_LOW = float(os.environ.get('SLOW_SLEEP_LOW', '0.032'))
SLOW_SLEEP_HIGH = float(os.environ.get('SLOW_SLEEP_HIGH', '0.048'))

# 论文设置：τ = ⌈T_max/T_min⌉，S = |邻居| - ⌈0.25*|邻居|⌉
# 如果未通过环境变量指定 MAX_LAG，则按论文公式计算：tau = ceil(SLOW_SLEEP / NORMAL_SLEEP)
if 'MAX_LAG' not in os.environ:
    tau = max(1, int(np.ceil(SLOW_SLEEP / max(NORMAL_SLEEP, 1e-6))))

# S_MODE 支持:
#   "formula_075" -> s_i = ceil(0.75 * |N_i|)   (per-node, based on topology)
#   "formula_alpha" -> s_i = ceil(ALPHA_DEP * |N_i|)
#   "fixed"       -> s_i = MIN_DEP_SET env value (global)
#   default       -> s_i = |N_i| - ceil(0.25*|N_i|)
_s_mode = os.environ.get("S_MODE", "default")
_alpha_dep = float(os.environ.get("ALPHA_DEP", "0.75"))
num_straggler_neighbors = max(1, int(np.ceil(0.25 * len(neighbors))))
if _s_mode == "formula_075":
    S = max(1, int(np.ceil(0.75 * len(neighbors))))
elif _s_mode == "formula_alpha":
    S = max(1, int(np.ceil(_alpha_dep * len(neighbors))))
elif 'MIN_DEP_SET' in os.environ:
    S = int(os.environ.get("MIN_DEP_SET"))
    S = min(S, len(neighbors))
else:
    S = max(1, len(neighbors) - num_straggler_neighbors)

try:
    FZ = pd.read_excel(io, sheet_name='城市参数')
    QY = FZ.values[0:sum(J), 1:9]

    DK_data = pd.read_excel(io, sheet_name='带宽上限')
    DKQ = DK_data.values[0:I, 1:I + 1]

    DK_data_cost = pd.read_excel(io, sheet_name='单价系数')
    DKC = DK_data_cost.values[0:I, 1:I + 1]

    # 负荷不平衡缩放：只缩放需求(c_in,w,c_out)，容量不变
    _demand_scale_env = os.environ.get("DEMAND_SCALE_JSON", "")
    if _demand_scale_env:
        import json as _json2
        _dscale = _json2.loads(_demand_scale_env)
        for k in range(I):
            factor = float(_dscale.get(str(k), 1.0))
            if factor != 1.0:
                row_start = sum(J[:k])
                row_end = row_start + J[k]
                QY[row_start:row_end, 0:3] *= factor  # 只缩放 c_in, w, c_out
        if bool(int(os.environ.get("ROUND_SCALED_DEMAND", "0"))):
            QY[:, 0:3] = np.rint(QY[:, 0:3])
        if rank == 0:
            for k in range(I):
                rows = QY[sum(J[:k]):sum(J[:k])+J[k]]
                d = rows[:, 0].sum() + rows[:, 1].sum() + rows[:, 2].sum()
                q = rows[:, 3].sum()
                print(f"  Region {k}: demand={d:.0f}, Q_cap={q:.0f}, factor={float(_dscale.get(str(k),1.0)):.2f}")

    if rank == 0:
        print(f"数据集加载成功: {dataset_name}, 文件: {io}, 节点数: {I}")
except Exception as e:
    if rank == 0:
        print(f"数据读取失败，请检查路径。错误信息: {e}")
    comm.Abort()

# 是否输出迭代日志
verbose_iter_log = bool(int(os.environ.get("VERBOSE_ITER_LOG", "0")))

# ================= 2. 异步 / ADMM 参数 =================
max_iter = int(os.environ.get("MAX_ITER", "3500"))

# S is already computed above based on S_MODE / MIN_DEP_SET / default formula

# maximum lagging level
if 'MAX_LAG' in os.environ:
    tau = int(os.environ.get("MAX_LAG"))

# 异步场景：
# 0 = 固定掉队者（可扩展性实验）；1 = 固定两节点慢（对照实验）
# 2 = 通畅网络+随机抖动；3 = 阻塞网络+同相位周期性突发；其他 = 轻微固定延迟
scenario = int(os.environ.get("SCENARIO", "1"))

# 动态阈值
conv_eps_abs = float(os.environ.get("CONV_EPS_ABS", "0.01")) 
conv_eps_rel = float(os.environ.get("CONV_EPS_REL", "0.01")) 
conv_stable_need = int(os.environ.get("CONV_STABLE_NEED", "5"))

# 近端项比例：gamma_prox = para * rho
para = float(os.environ.get("PROX_RATIO", "1.5"))

# rho 更新策略：
# fixed / adaptive / hybrid / staged_fixed
# 辅助变量版本建议优先使用 fixed，因为 residual balancing 不再是标准 ADMM 残差平衡
rho_strategy = os.environ.get("RHO_STRATEGY", "fixed")
rho0 = float(os.environ.get("RHO0", "200"))
rho_fixed_value = float(os.environ.get("RHO_FIXED", "160"))
hybrid_switch_iter = int(os.environ.get("HYBRID_SWITCH_ITER", "10"))
rho_stage1_value = float(os.environ.get("RHO_STAGE1", "500"))
rho_stage2_value = float(os.environ.get("RHO_STAGE2", "18"))
rho_stage_switch_iter = int(os.environ.get("RHO_STAGE_SWITCH_ITER", "20"))

# 自适应 rho 参数
mu = float(os.environ.get("RHO_MU", "8"))
tau_inc = float(os.environ.get("RHO_TAU_INC", "1.2"))
tau_dec = float(os.environ.get("RHO_TAU_DEC", "1.2"))
rho_min = float(os.environ.get("RHO_MIN", "1e-3"))
rho_max = float(os.environ.get("RHO_MAX", "500"))

if rho_strategy not in ("fixed", "adaptive", "hybrid", "staged_fixed"):
    if rank == 0:
        print(f"未知 rho_strategy: {rho_strategy}，可选 fixed / adaptive / hybrid / staged_fixed")
    comm.Abort()

if rho_strategy == "fixed":
    rho = rho_fixed_value
elif rho_strategy == "staged_fixed":
    rho = rho_stage1_value
else:
    rho = min(max(rho0, rho_min), rho_max)

if rank == 0:
    _topo_label = "full" if _topo_env == "full" else "sparse"
    print(f"异步 ADMM 参数: max_iter={max_iter}, S={S}, tau={tau}, scenario={scenario}, "
          f"topology={_topo_label}, |N_i|={len(neighbors)}, S_mode={_s_mode}")
    if rho_strategy == "fixed":
        print(f"rho 策略: 全程固定, rho={rho_fixed_value}")
    elif rho_strategy == "adaptive":
        print("rho 策略: 全程自适应；注意辅助变量版本中该策略不再对应标准残差平衡")
    elif rho_strategy == "staged_fixed":
        print(
            f"rho 策略: staged_fixed, 前段 rho={rho_stage1_value}, "
            f"switch_iter={rho_stage_switch_iter}, 后段 rho={rho_stage2_value}"
        )
    else:
        print(
            f"rho 策略: hybrid, switch_iter={hybrid_switch_iter}, "
            f"切换后固定 rho={rho_fixed_value}"
        )

# Gurobi 环境
global_env = gp.Env(empty=True)
global_env.setParam("OutputFlag", 0)
_grb_threads = int(os.environ.get("GRB_THREADS", "0"))
if _grb_threads > 0:
    global_env.setParam("Threads", _grb_threads)
global_env.start()


# ================= 3. 本地子问题求解器：辅助变量版本 =================
def compute_x(k, rho, gamma_prox, x_prev_local, u_in, env):
    r"""
    辅助变量版本的本地子问题。

    u_in[j] 表示 scheduler j 发给 scheduler k 的辅助变量 u_{jk}。
    在本代码中，u_in[j] 已经按照 scheduler k 的本地变量顺序存储：
        u_in[j][0] 对应 x[k, j]
        u_in[j][1] 对应 x[j, k]

    本地目标中的辅助变量项为：
        u_{jk}^T x_{kj}^{(k)} + rho/2 * ||x_{kj}^{(k)}||_2^2

    因此不再需要显式接收邻居 primal copy，也不再显式维护 lambda。
    """
    model = gp.Model(f"Subproblem_{k}", env=env)

    # 仅对邻居定义边变量
    x_ij = model.addVars(neighbors, lb=0, vtype=GRB.INTEGER, name='x_ij')  # x[k,j]
    x_ji = model.addVars(neighbors, lb=0, vtype=GRB.INTEGER, name='x_ji')  # x[j,k]

    c_in = model.addVars(J[k], lb=0, vtype=GRB.INTEGER, name='c_in')
    c_out = model.addVars(J[k], lb=0, vtype=GRB.INTEGER, name='c_out')
    w = model.addVars(J[k], J[k], lb=0, vtype=GRB.INTEGER, name='w')
    c = model.addVars(J[k], J[k], lb=0, vtype=GRB.INTEGER, name='c')
    Q = model.addVars(J[k], lb=0, vtype=GRB.INTEGER, name='Q')
    b = model.addVars(J[k], lb=0, vtype=GRB.INTEGER, name='b')

    # ========== 物理约束 ==========
    model.addConstr(x_ji.sum() == c_in.sum(), name='(3-2-1)')
    model.addConstr(x_ij.sum() == c_out.sum(), name='(3-2-2)')

    model.addConstrs(
        (0.25 * 10000 * (x_ij[j] + x_ji[j]) <= 2**20 * DKQ[k][j] for j in neighbors),
        name='(3-6-1)'
    )

    model.addConstrs(
        (0.25 * 10000 * (c.sum('*', m) + c.sum(m, '*') - 2 * c[m, m] + c_in[m] + c_out[m]) +
         0.3 * 10000 * (w.sum('*', m) + w.sum(m, '*') - 2 * w[m, m]) == b[m]
         for m in range(J[k])),
        name='(3-6-2)'
    )

    model.addConstrs(
        (b[m] <= 2 ** 20 * QY[sum(J[:k]) + m, 6] for m in range(J[k])),
        name='(3-6-3)'
    )

    model.addConstrs(
        (Q[m] <= QY[sum(J[:k]) + m, 4] for m in range(J[k])),
        name='(3-5-2)'
    )

    model.addConstrs(
        (w.sum(m, '*') == QY[sum(J[:k]) + m, 2] for m in range(J[k])),
        name='(3-3-1)'
    )

    model.addConstrs(
        (c.sum(m, '*') + c_out[m] == QY[sum(J[:k]) + m, 1] for m in range(J[k])),
        name='(3-4-1)'
    )

    model.addConstrs(
        (QY[sum(J[:k]) + m, 1] + QY[sum(J[:k]) + m, 2] + QY[sum(J[:k]) + m, 3] +
         c.sum('*', m) - c.sum(m, '*') + c_in[m] - c_out[m] +
         w.sum('*', m) - w.sum(m, '*') == Q[m]
         for m in range(J[k])),
        name='(3-5-1)'
    )

    # ========== 物理成本 ==========
    CONST_COST = 0.5 * 0.25 * 10000 * 2000 / (2**10)

    k1 = gp.quicksum(
        (x_ij[j] + x_ji[j]) * DKC[k][j] * CONST_COST
        for j in neighbors
    )

    k2 = gp.quicksum(
        2000 / 2**10 * b[m] * QY[sum(J[:k]) + m, 7] +
        Q[m] * QY[sum(J[:k]) + m, 5] * 45 * 24 * 365 * 10 * 0.1
        for m in range(J[k])
    )

    # ========== 辅助变量 ADMM 项 ==========
    aux_term = gp.LinExpr()
    prox_term = gp.LinExpr()

    for j in neighbors:
        u0 = float(u_in[j][0])  # 对应 x[k,j]
        u1 = float(u_in[j][1])  # 对应 x[j,k]

        aux_term += u0 * x_ij[j] + (rho / 2.0) * x_ij[j] * x_ij[j]
        aux_term += u1 * x_ji[j] + (rho / 2.0) * x_ji[j] * x_ji[j]

        # 近端项，用于整数实现中的稳定化
        prox_term += (gamma_prox / 2.0) * (x_ij[j] - x_prev_local[k, j]) * (x_ij[j] - x_prev_local[k, j])
        prox_term += (gamma_prox / 2.0) * (x_ji[j] - x_prev_local[j, k]) * (x_ji[j] - x_prev_local[j, k])

    model.setObjective(k1 + k2 + aux_term + prox_term, GRB.MINIMIZE)
    model.optimize()

    ans_x = np.zeros((I, I))
    if model.status == GRB.OPTIMAL:
        true_obj = k1.getValue() + k2.getValue()
        for j in neighbors:
            ans_x[k, j] = x_ij[j].X
            ans_x[j, k] = x_ji[j].X
        return ans_x, true_obj
    else:
        raise RuntimeError(f"[Rank {k}] Gurobi failed with status {model.status}")


def workload(iter_k=0):
    if scenario ==1:  # CASE 1_fixed and CASE 2_SENSITIVITY
        if rank==2:
            time.sleep(SLOW_SLEEP)
        elif rank==5:
            time.sleep(SLOW_SLEEP)
            time.sleep(SLOW_SLEEP)
        else:
            time.sleep(NORMAL_SLEEP)
    elif scenario == 0:  # 通畅网络：全程正常速度 + 随机抖动
        jitter = abs(np.random.normal(0.0, CONGESTION_JITTER_STD))
        time.sleep(NORMAL_SLEEP + jitter)
    elif scenario == 2:  # 固定基础时延 + 对数正态抖动：异构节点 + 随机网络抖动
        # 每个 rank 有固定的基础延迟和放大系数，模拟计算资源异构
        _base   = [0.004, 0.005, 0.008, 0.006, 0.007, 0.020, 0.005, 0.009]
        _factor = [1.0,   1.0,   1.5,   1.1,   1.2,   2.0,   1.0,   1.6  ]
        my_base   = _base[rank % len(_base)]
        my_factor = _factor[rank % len(_factor)]
        jitter = np.random.lognormal(mean=LOGNORMAL_MEAN, sigma=LOGNORMAL_SIGMA)
        time.sleep(my_factor * (my_base + jitter))
    elif scenario == 4:  # CASE 3_SCALIABILITY
        if rank in slow_ranks:
            time.sleep(SLOW_SLEEP)
        else:
            time.sleep(NORMAL_SLEEP)
    elif scenario == 5:  # 固定掉队者基础时延 + 随机相对扰动
        base_delay = SLOW_SLEEP if rank in slow_ranks else NORMAL_SLEEP
        jitter = base_delay * abs(np.random.normal(0.0, RANDOM_DELAY_CV))
        time.sleep(base_delay + jitter)
    elif scenario == 6:  # 固定掉队者身份 + 均匀随机更新时间
        if rank in slow_ranks:
            time.sleep(np.random.uniform(SLOW_SLEEP_LOW, SLOW_SLEEP_HIGH))
        else:
            time.sleep(np.random.uniform(NORMAL_SLEEP_LOW, NORMAL_SLEEP_HIGH))
    else:
        time.sleep(NORMAL_SLEEP)


def pack_u_for_receiver(u_local_order):
    """
    将本节点本地顺序下的 u_out[j] 打包为接收方本地顺序。

    本节点 rank 对邻居 j 的本地变量顺序是：
        [x[rank,j], x[j,rank]]

    邻居 j 接收后，其本地变量顺序是：
        [x[j,rank], x[rank,j]]

    因此发送前需要交换两个分量。
    """
    return np.array([u_local_order[1], u_local_order[0]], dtype=float)


# ================= 4. 主循环 =================
if __name__ == '__main__':
    local_k = 0
    status = MPI.Status()
    wall_start = timer()

    # 本节点的局部 primal 决策矩阵
    # 对边 {rank,j}，本节点维护两个方向变量：x[rank,j] 与 x[j,rank]
    x_local = np.zeros((I, I))

    # u_in[j]：邻居 j 发给本节点 rank 的辅助变量 u_{j,rank}
    # 按本节点本地变量顺序存储：
    #   u_in[j][0] 对应 x[rank,j]
    #   u_in[j][1] 对应 x[j,rank]
    _u_init_mode = os.environ.get("U_INIT_MODE", "zero")
    _u_init_scale = float(os.environ.get("U_INIT_SCALE", "1.0"))
    if _u_init_mode == "zero":
        u_in = {j: np.zeros(2, dtype=float) for j in neighbors}
        u_out = {j: np.zeros(2, dtype=float) for j in neighbors}
    else:
        _rng = np.random.RandomState(42 + rank)
        _sigma = _u_init_scale * rho
        u_in = {j: _rng.normal(0, _sigma, size=2).astype(float) for j in neighbors}
        u_out = {j: _rng.normal(0, _sigma, size=2).astype(float) for j in neighbors}

    # 时间戳缓存：记录来自邻居的最近轮次
    last_t = {j: -1 for j in neighbors}
    lag = {j: 0 for j in neighbors}

    # 残差与指标记录
    err_pri = np.zeros(max_iter + 5)   # 辅助变量版本中表示 dx = ||x^{k+1}-x^k||
    err_dual = np.zeros(max_iter + 5)  # 辅助变量版本中表示 du = ||u^{k+1}-u^k||
    eps_pri_hist = np.zeros(max_iter + 5)
    eps_dual_hist = np.zeros(max_iter + 5)
    obj_local_hist = np.zeros(max_iter + 5)
    iter_wall_hist = np.zeros(max_iter + 5)
    final_phys_obj = 0.0

    time_comm = 0.0
    time_comp = 0.0
    msg_send_cnt = 0
    msg_recv_cnt = 0
    bytes_send = 0
    bytes_recv = 0

    # 停止传播机制
    local_converged = 0
    local_stop = 0
    stable_cnt = 0
    neighbor_stop = {j: 0 for j in neighbors}

    in_fixed_phase = (rho_strategy == "fixed")
    hybrid_switched = False
    rho_freeze_value = rho if rho_strategy == "fixed" else None

    # 异步发送请求池
    pending_sends = []

    while local_k < max_iter:
        # 清理已经完成的异步发送请求，避免列表无限增长
        alive_reqs = []
        for req in pending_sends:
            if not req.Test():
                alive_reqs.append(req)
        pending_sends = alive_reqs

        # ---------- Step 1: 广播本地最新辅助变量 u_out ----------
        tic = timer()
        for dest in neighbors:
            u_payload = pack_u_for_receiver(u_out[dest])
            msg = {
                "t": local_k,
                "u_msg": u_payload.tolist(),
                "stop": local_stop,
            }

            # 模拟消息丢失，但不丢弃 stop 控制消息
            if not (scenario == 3 and local_stop == 0 and local_k < max_iter - 1 and np.random.rand() < 0.05):
                pending_sends.append(comm.isend(copy.deepcopy(msg), dest=dest, tag=0))
                msg_send_cnt += 1
                bytes_send += len(pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL))

        # ---------- Step 2: 接收消息，直到满足 minimum dependency set 和 lag 限制 ----------
        fresh_received = set()
        got_new = False

        while True:
            got_new = False

            # 非阻塞探测 + 批量读取 + 最新时间戳覆盖
            while comm.Iprobe(source=MPI.ANY_SOURCE, tag=0, status=status):
                src = status.Get_source()
                data = comm.recv(source=src, tag=0, status=status)
                msg_recv_cnt += 1
                bytes_recv += len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
                got_new = True

                stop_recv = int(data.get("stop", 0))
                if stop_recv == 1:
                    neighbor_stop[src] = 1

                t_recv = int(data["t"])
                u_recv = np.array(data["u_msg"], dtype=float)

                # 只接受更新鲜的消息
                if t_recv > last_t[src]:
                    last_t[src] = t_recv

                    # 发送方已经交换过分量，因此这里可以直接按本节点顺序存储
                    # u_in[src][0] 对应 x[rank,src]
                    # u_in[src][1] 对应 x[src,rank]
                    u_in[src] = u_recv

                    if neighbor_stop[src] == 0:
                        fresh_received.add(src)

            active_neighbors = [j for j in neighbors if neighbor_stop[j] == 0]

            # 更新 lag；已停止邻居不计入 lag
            for j in neighbors:
                if neighbor_stop[j] == 0:
                    if last_t[j] >= 0:
                        lag[j] = local_k - last_t[j]
                    else:
                        lag[j] = local_k + 1
                else:
                    lag[j] = 0

            fresh_active = [j for j in fresh_received if neighbor_stop[j] == 0]
            required_fresh = min(S, len(active_neighbors))
            enough_neighbors = (len(fresh_active) >= required_fresh)
            lag_ok = (max(lag[j] for j in active_neighbors) <= tau) if len(active_neighbors) > 0 else True

            if (enough_neighbors and lag_ok) or (len(active_neighbors) == 0):
                break

            # 如果没有新消息，则阻塞等待，避免 CPU 空转
            if not got_new:
                comm.Probe(source=MPI.ANY_SOURCE, tag=0, status=status)

        toc = timer()
        time_comm += toc - tic

        # ---------- Step 3: 如果本地已经宣布停止，则检查邻居状态 ----------
        if local_stop == 1:
            if sum(neighbor_stop.values()) == len(neighbors):
                if verbose_iter_log:
                    print(f"Rank {rank} | All neighbors stopped, safe exit.")
                break

            # 如果已经停止但收到活跃邻居的新消息，则唤醒
            if got_new:
                local_stop = 0
                stable_cnt = 0
                if verbose_iter_log:
                    print(f"Rank {rank} | WAKE UP by incoming u messages.")
            else:
                if verbose_iter_log:
                    print(
                        f"Rank {rank} | Passive wait | "
                        f"active_neighbors={len([j for j in neighbors if neighbor_stop[j] == 0])} | "
                        f"all_neighbor_stop={sum(neighbor_stop.values())}/{len(neighbors)}"
                    )
                time.sleep(0.001)
                continue

        # ---------- Step 4: 本地优化 ----------
        tic = timer()
        workload(local_k)

        gamma_prox = para * rho
        x_prev_local = copy.deepcopy(x_local)
        u_out_old = copy.deepcopy(u_out)

        x_new, phys_obj = compute_x(
            rank,
            rho,
            gamma_prox,
            x_prev_local,
            u_in,
            global_env
        )

        obj_local_hist[local_k] = phys_obj
        final_phys_obj = phys_obj

        # ---------- Step 5: 辅助变量更新 + 固定点残差 ----------
        x_change_sq = 0.0
        u_change_sq = 0.0
        norm_x_sq = 0.0
        norm_u_sq = 0.0

        for j in neighbors:
            # 本节点 rank 在边 {rank,j} 上的本地变量向量，顺序为：
            # [x[rank,j], x[j,rank]]
            x_vec_new = np.array(
                [x_new[rank, j], x_new[j, rank]],
                dtype=float
            )
            x_vec_old = np.array(
                [x_prev_local[rank, j], x_prev_local[j, rank]],
                dtype=float
            )

            # 紧凑递推：
            # u_{ij}^{k+1} = (u_{ij}^{k} - u_{ji}^{k}) / 2 - rho * x_{ij}^{(i),k+1}
            #
            # u_out_old[j] 和 u_in[j] 均已按本节点本地变量顺序排列。
            u_out[j] = 0.5 * (u_out_old[j] - u_in[j]) - rho * x_vec_new

            x_change_sq += np.sum((x_vec_new - x_vec_old) ** 2)
            u_change_sq += np.sum((u_out[j] - u_out_old[j]) ** 2)

            norm_x_sq += np.sum(x_vec_new ** 2)
            norm_u_sq += np.sum(u_out[j] ** 2)

        err_pri[local_k] = np.sqrt(x_change_sq)
        err_dual[local_k] = np.sqrt(u_change_sq)

        # 动态阈值：辅助变量版本的 dx / du 固定点判据
        n_vars = 2 * len(neighbors)
        eps_pri_cur = np.sqrt(n_vars) * conv_eps_abs + conv_eps_rel * max(np.sqrt(norm_x_sq), 1.0)
        eps_dual_cur = np.sqrt(n_vars) * conv_eps_abs + conv_eps_rel * max(np.sqrt(norm_u_sq), 1.0)

        eps_pri_hist[local_k] = eps_pri_cur
        eps_dual_hist[local_k] = eps_dual_cur

        x_local = copy.deepcopy(x_new)

        toc = timer()
        time_comp += toc - tic
        iter_wall_hist[local_k] = timer() - wall_start

        # ---------- Step 6: 本地停止判定 ----------
        local_converged = int(
            (err_pri[local_k] <= eps_pri_cur) and
            (err_dual[local_k] <= eps_dual_cur)
        )

        if local_converged == 1:
            stable_cnt += 1
        else:
            stable_cnt = 0
            local_stop = 0

        if local_converged == 1 and stable_cnt >= conv_stable_need:
            local_stop = 1

        # ---------- Step 7: rho 更新 ----------
        # 注意：辅助变量版本中 err_pri / err_dual 分别为 dx / du，
        # 不再严格等价于标准 ADMM primal / dual residual。
        local_pri = err_pri[local_k]
        local_dual = err_dual[local_k]

        if rho_strategy == "fixed":
            rho = rho_fixed_value
            in_fixed_phase = True

        elif rho_strategy == "staged_fixed":
            if local_k < rho_stage_switch_iter:
                rho = rho_stage1_value
            else:
                rho = rho_stage2_value
            in_fixed_phase = True

        elif rho_strategy == "hybrid":
            if (not hybrid_switched) and (local_k >= hybrid_switch_iter):
                hybrid_switched = True
                rho_freeze_value = rho_fixed_value

            if hybrid_switched:
                if rho_freeze_value is None:
                    rho_freeze_value = rho_fixed_value
                rho = rho_freeze_value
                in_fixed_phase = True
            else:
                in_fixed_phase = False
                if local_pri > mu * local_dual:
                    rho = min(rho * tau_inc, rho_max)
                elif local_dual > mu * local_pri:
                    rho = max(rho / tau_dec, rho_min)

        else:  # adaptive
            in_fixed_phase = False
            if local_pri > mu * local_dual:
                rho = min(rho * tau_inc, rho_max)
            elif local_dual > mu * local_pri:
                rho = max(rho / tau_dec, rho_min)

        if verbose_iter_log and rank == 0:
            print(
                f"Rank {rank} | Iter {local_k+1:04d} | Obj {phys_obj:.2f} | "
                f"dx {err_pri[local_k]:.4e} | du {err_dual[local_k]:.4e} | "
                f"eps_x {eps_pri_cur:.4e} | eps_u {eps_dual_cur:.4e} | "
                f"fresh {len(fresh_active)}/{required_fresh} | "
                f"maxlag {max([lag[j] for j in neighbors], default=0)} | "
                f"rho {rho:.4f} | fixed_phase {int(in_fixed_phase)} | "
                f"stable {stable_cnt}/{conv_stable_need} | stop {local_stop}/{sum(neighbor_stop.values())}"
            )

        local_k += 1

        # 如果自己已停且邻居都已停，安全退出
        if local_stop == 1 and sum(neighbor_stop.values()) == len(neighbors):
            break

    # ================= 5. 结束前再广播几次 stop，帮助邻居退出 =================
    local_stop = 1
    if local_stop == 1:
        for _ in range(3):
            for dest in neighbors:
                u_payload = pack_u_for_receiver(u_out[dest])
                msg_final = {
                    "t": local_k,
                    "u_msg": u_payload.tolist(),
                    "stop": 1,
                }
                pending_sends.append(comm.isend(copy.deepcopy(msg_final), dest=dest, tag=0))
                msg_send_cnt += 1
                bytes_send += len(pickle.dumps(msg_final, protocol=pickle.HIGHEST_PROTOCOL))
            time.sleep(0.01)

    # 等待剩余发送完成，避免过早 Finalize 导致消息丢失
    if len(pending_sends) > 0:
        MPI.Request.Waitall(pending_sends)

    # ================= 6. 汇总 =================
    valid_len = max(local_k, 1)
    valid_pri = err_pri[:valid_len]
    valid_dual = err_dual[:valid_len]
    valid_eps_pri = eps_pri_hist[:valid_len]
    valid_eps_dual = eps_dual_hist[:valid_len]
    valid_obj_local = obj_local_hist[:valid_len]
    valid_iter_wall = iter_wall_hist[:valid_len]

    all_pri = comm.gather(valid_pri, root=0)
    all_dual = comm.gather(valid_dual, root=0)
    all_eps_pri = comm.gather(valid_eps_pri, root=0)
    all_eps_dual = comm.gather(valid_eps_dual, root=0)
    all_obj_local = comm.gather(valid_obj_local.tolist(), root=0)
    all_iter_wall = comm.gather(valid_iter_wall.tolist(), root=0)
    all_final_obj = comm.gather(final_phys_obj, root=0)
    all_iters = comm.gather(local_k, root=0)
    all_time_comm = comm.gather(time_comm, root=0)
    all_time_comp = comm.gather(time_comp, root=0)
    all_msg_send_cnt = comm.gather(msg_send_cnt, root=0)
    all_msg_recv_cnt = comm.gather(msg_recv_cnt, root=0)
    all_bytes_send = comm.gather(bytes_send, root=0)
    all_bytes_recv = comm.gather(bytes_recv, root=0)
    all_x_local = comm.gather(x_local, root=0)

    local_region_price = [
        float(v) for v in QY[sum(J[:rank]):sum(J[:rank]) + J[rank], 5].tolist()
    ]
    all_region_price = comm.gather(local_region_price, root=0)

    wall_total = timer() - wall_start
    all_wall = comm.gather(wall_total, root=0)

    if rank == 0:
        total_final_obj = float(np.sum(all_final_obj))
        avg_iter = float(np.mean(all_iters))
        wall_clock_total = float(np.max(all_wall))
        node_avg_runtime = float(np.mean(all_wall))
        avg_comp_time = float(np.mean(all_time_comp))
        avg_comm_time = float(np.mean(all_time_comm))

        abs_diff = float('nan')
        rel_diff = float('nan')
        if centralized_base is not None:
            abs_diff = abs(total_final_obj - centralized_base)
            rel_diff = abs_diff / max(abs(centralized_base), 1.0)

        # 离线共识检查：不参与迭代通信，仅用于最终诊断
        # 对每条有向变量 x[i,j]，不同 rank 本地副本可能存储在两个节点的 x_local 中。
        # 对边 {i,j}：
        #   rank i 的本地 x_local[i,j] 与 rank j 的本地 x_local[i,j] 应趋于一致；
        #   rank i 的本地 x_local[j,i] 与 rank j 的本地 x_local[j,i] 应趋于一致。
        offline_consensus_sq = 0.0
        if _topo_env == "full":
            _edges = [(i, j) for i in range(size) for j in range(i + 1, size)]
        else:
            _td = _json.loads(_topo_env)
            _edges = [(i, j) for i in range(size) for j in _td[str(i)] if j > i]
        for i, j in _edges:
            xi = all_x_local[i]
            xj = all_x_local[j]
            offline_consensus_sq += (xi[i, j] - xj[i, j]) ** 2
            offline_consensus_sq += (xi[j, i] - xj[j, i]) ** 2
        offline_consensus_res = float(np.sqrt(offline_consensus_sq))
        comm_ratio = avg_comm_time / (avg_comp_time + avg_comm_time) if (avg_comp_time + avg_comm_time) > 0 else float('nan')

        K_max = float(np.max(all_iters))
        K_min = float(np.min(all_iters))
        K_bar = float(np.mean(all_iters))
        iter_balance = (K_max - K_min) / K_bar * 100.0 if K_bar > 0 else 0.0

        print("\n>>> 异步辅助变量 ADMM 核心指标（按对比要求）")
        print(f">>> 目标函数值（总）: {total_final_obj:.6f}")
        print(f">>> 与集中式偏差（绝对）: {abs_diff:.6f}")
        print(f">>> 与集中式偏差（相对）: {rel_diff:.6%}" if not np.isnan(rel_diff) else ">>> 与集中式偏差（相对）: nan")
        print(f">>> 总耗时 wall-clock: {wall_clock_total:.6f}s")
        print(f">>> 节点平均运行时间: {node_avg_runtime:.6f}s")
        print(f">>> 平均计算时间: {avg_comp_time:.6f}s")
        print(f">>> 平均通信时间（实测）: {avg_comm_time:.6f}s")
        print(f">>> 通信时间占比: {comm_ratio:.2%}")
        print(f">>> 平均迭代次数: {avg_iter:.3f}")
        print(f">>> 平均发送消息数: {np.mean(all_msg_send_cnt):.2f}")
        print(f">>> 平均发送字节数: {np.mean(all_bytes_send):.2f} B")
        print(f">>> 迭代不平衡度: {iter_balance:.2f}% (K_max={K_max:.0f}, K_min={K_min:.0f}, K_bar={K_bar:.2f})")
        print(f">>> 离线 primal consensus residual: {offline_consensus_res:.6e}")

        if verbose_iter_log:
            print("\n>>> 各 rank 区域电价")
            for r in range(size):
                price_list = all_region_price[r]
                avg_price = float(np.mean(price_list)) if len(price_list) > 0 else float('nan')
                print(f">>> Rank {r}: prices={price_list}, avg={avg_price:.6f}")

        print("\n>>> 开始汇总并保存数据...")
        try:
            async_trace_path = os.path.join(base_dir, 'admm_async_auxu_obj_trace.csv')
            trace_recs = []
            for r in range(size):
                trace_r = all_obj_local[r]
                for it, val in enumerate(trace_r):
                    trace_recs.append({'rank': r, 'iter': it + 1, 'obj_local': val})
            pd.DataFrame(trace_recs).to_csv(async_trace_path, index=False, encoding='utf-8-sig')
            print(f">>> 异步各节点目标值轨迹已保存至: {async_trace_path}")

            # 汇总全局 trace：若某 rank 已停，则顺延最后值
            max_len = max(len(tr) for tr in all_obj_local)
            dx_global = np.zeros(max_len)
            du_global = np.zeros(max_len)
            eps_x_global = np.zeros(max_len)
            eps_u_global = np.zeros(max_len)
            norm_dx_global = np.zeros(max_len)
            norm_du_global = np.zeros(max_len)
            obj_total = np.zeros(max_len)
            wall_time_global = np.zeros(max_len)

            for i in range(max_len):
                p_v = [all_pri[r][i] if i < len(all_pri[r]) else all_pri[r][-1] for r in range(size)]
                d_v = [all_dual[r][i] if i < len(all_dual[r]) else all_dual[r][-1] for r in range(size)]
                ep_v = [all_eps_pri[r][i] if i < len(all_eps_pri[r]) else all_eps_pri[r][-1] for r in range(size)]
                ed_v = [all_eps_dual[r][i] if i < len(all_eps_dual[r]) else all_eps_dual[r][-1] for r in range(size)]
                o_v = [all_obj_local[r][i] if i < len(all_obj_local[r]) else all_obj_local[r][-1] for r in range(size)]
                w_v = [all_iter_wall[r][i] if i < len(all_iter_wall[r]) else all_iter_wall[r][-1] for r in range(size)]

                dx_global[i] = np.max(p_v)
                du_global[i] = np.max(d_v)
                eps_x_global[i] = np.max(ep_v)
                eps_u_global[i] = np.max(ed_v)
                norm_dx_global[i] = np.max([
                    p_v[r] / max(ep_v[r], 1e-12) for r in range(size)
                ])
                norm_du_global[i] = np.max([
                    d_v[r] / max(ed_v[r], 1e-12) for r in range(size)
                ])
                obj_total[i] = np.sum(o_v)
                wall_time_global[i] = np.max(w_v)

            df_trace_async = pd.DataFrame({
                'iter': np.arange(1, max_len + 1),
                'wall_time_global': wall_time_global,
                'dx_global': dx_global,
                'du_global': du_global,
                'eps_x_global': eps_x_global,
                'eps_u_global': eps_u_global,
                'norm_dx_global': norm_dx_global,
                'norm_du_global': norm_du_global,
                'obj_global': obj_total,
            })
            if S == 0 and tau == 16000:
                async_trace_filename = 'admm_trace_async_undelay.csv'
            elif S == 0 and tau == 16:
                async_trace_filename = 'admm_trace_async_delay.csv'
            else:
                async_trace_filename = 'admm_trace_async.csv'

            async_global_trace = os.path.join(base_dir, async_trace_filename)
            df_trace_async.to_csv(async_global_trace, index=False, encoding='utf-8-sig')
            print(f">>> 异步辅助变量全局收敛轨迹已保存至: {async_global_trace}")

            summary_path = os.path.join(base_dir, 'admm_async_auxu_rank_summary.csv')
            df_summary = pd.DataFrame({
                'rank': np.arange(size),
                'iter': all_iters,
                'final_obj_local': all_final_obj,
                'time_comm': all_time_comm,
                'time_comp': all_time_comp,
                'wall_time': all_wall,
                'msg_send_cnt': all_msg_send_cnt,
                'msg_recv_cnt': all_msg_recv_cnt,
                'bytes_send': all_bytes_send,
                'bytes_recv': all_bytes_recv,
            })
            df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
            print(f">>> 异步辅助变量 rank 汇总 CSV 已保存至: {summary_path}")

            metrics_path = os.path.join(base_dir, 'admm_async_auxu_metrics.csv')
            df_metrics = pd.DataFrame([{
                'obj_total': total_final_obj,
                'abs_diff_to_centralized': abs_diff,
                'rel_diff_to_centralized': rel_diff,
                'wall_clock_total': wall_clock_total,
                'node_avg_runtime': node_avg_runtime,
                'avg_comp_time': avg_comp_time,
                'avg_comm_time': avg_comm_time,
                'avg_iter': avg_iter,
                'avg_msg_send_cnt': float(np.mean(all_msg_send_cnt)),
                'avg_bytes_send': float(np.mean(all_bytes_send)),
                'iter_balance_pct': iter_balance,
                'K_max': K_max,
                'K_min': K_min,
                'offline_primal_consensus_residual': offline_consensus_res,
            }])
            df_metrics.to_csv(metrics_path, index=False, encoding='utf-8-sig')
            print(f">>> 异步辅助变量核心指标 CSV 已保存至: {metrics_path}")

            # 保存最终本地 x，用于调试比较
            final_x_path = os.path.join(base_dir, 'admm_async_auxu_final_x.pkl')
            with open(final_x_path, 'wb') as f:
                pickle.dump(all_x_local, f)
            print(f">>> 最终各 rank 本地 x 已保存至: {final_x_path}")

        except Exception as e:
            print(f"保存失败: {e}")

    print(f"[Rank {rank}] 通信耗时: {time_comm:.3f}s | 计算耗时: {time_comp:.3f}s")

    comm.Barrier()
    try:
        MPI.Finalize()
    except Exception:
        pass
