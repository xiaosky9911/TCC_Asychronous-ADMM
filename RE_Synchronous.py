# -*- coding: utf-8 -*-
# 同步边共识 ADMM 算力调度
# 示例：
# 4节点 160 rho 策略："fixed" ，160；终止条件：primal/dual eps 0.01，稳定满足 5 轮
# 8节点 160 rho 策略："fixed" ，160；终止条件：primal/dual eps 10，稳定满足 5 轮
# mpiexec -np 4 python d:\yanjiusheng\ready\Cloud_TCC_Revision\RE_Synchronous.py need_data_N4
# mpiexec -np 8 python d:\yanjiusheng\ready\Cloud_TCC_Revision\RE_Synchronous.py need_data_N8
# $env:CENTRALIZED_BASE="1182823039.614968"; mpiexec -np 8 python d:\yanjiusheng\ready\Cloud_TCC_Revision\RE_Synchronous.py need_data_N8

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

base_dir = r'd:\yanjiusheng\ready\Cloud_TCC_Revision'

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

dataset_name = os.environ.get("DATASET_NAME", "need_data_N4")
if len(sys.argv) > 1:
    dataset_name = sys.argv[1]

# 可选：集中式基准值，用于自动对比相对误差
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

neighbors = [j for j in range(I) if j != rank]

try:
    FZ = pd.read_excel(io, sheet_name='城市参数')
    QY = FZ.values[0:sum(J), 1:9]
    DK_data = pd.read_excel(io, sheet_name='带宽上限')
    DKQ = DK_data.values[0:I, 1:I + 1]
    DK_data_cost = pd.read_excel(io, sheet_name='单价系数')
    DKC = DK_data_cost.values[0:I, 1:I + 1]
    if rank == 0:
        print(f"数据集加载成功: {dataset_name}, 文件: {io}, 节点数: {I}")
except Exception as e:
    if rank == 0:
        print(f"数据读取失败，请检查路径。错误信息: {e}")
    comm.Abort()

# ================= 2. 同步 / ADMM 参数 =================
max_iter = int(os.environ.get("MAX_ITER", "3500"))

conv_eps_abs = float(os.environ.get("CONV_EPS_ABS", "0.01"))
conv_eps_rel = float(os.environ.get("CONV_EPS_REL", "0.01"))
conv_stable_need = int(os.environ.get("CONV_STABLE_NEED", "5"))
para = float(os.environ.get("PROX_RATIO", "1.5"))
scenario = int(os.environ.get("SCENARIO", "1"))
verbose_iter_log = bool(int(os.environ.get("VERBOSE_ITER_LOG", "0")))

# rho 更新策略：fixed / adaptive / hybrid / staged_fixed
rho_strategy = os.environ.get("RHO_STRATEGY", "fixed")
rho0 = float(os.environ.get("RHO0", "400"))
rho_fixed_value = float(os.environ.get("RHO_FIXED", "160"))
hybrid_switch_iter = int(os.environ.get("HYBRID_SWITCH_ITER", "10"))
rho_stage1_value = float(os.environ.get("RHO_STAGE1", "500"))
rho_stage2_value = float(os.environ.get("RHO_STAGE2", "18"))
rho_stage_switch_iter = int(os.environ.get("RHO_STAGE_SWITCH_ITER", "20"))

# 自适应 rho 参数（残差平衡）
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
    # 统一裁剪到可接受范围，避免初值过大造成前期冲击
    rho = min(max(rho0, rho_min), rho_max)

if rank == 0:
    if rho_strategy == "fixed":
        print(f"rho 策略: 全程固定, rho={rho_fixed_value}")
    elif rho_strategy == "adaptive":
        print("rho 策略: 全程自适应")
    elif rho_strategy == "staged_fixed":
        print(
            f"rho 策略: staged_fixed, 前段 rho={rho_stage1_value}, "
            f"switch_iter={rho_stage_switch_iter}, 后段 rho={rho_stage2_value}"
        )
    else:
        print(
            f"rho 策略: hybrid, switch_iter={hybrid_switch_iter}, "
            f"切换后固定 rho=rho_fixed={rho_fixed_value}"
        )

# Gurobi 环境
global_env = gp.Env(empty=True)
global_env.setParam("OutputFlag", 0)
global_env.setParam("Threads", 1)
global_env.setParam("Seed", 1)
global_env.start()

# 掉队者配置（仅初始化一次）
SLOW_RATIO = float(os.environ.get('SLOW_RATIO', '0.25'))
SLOW_SLEEP = float(os.environ.get('SLOW_SLEEP', '0.04'))
NORMAL_SLEEP = float(os.environ.get('NORMAL_SLEEP', '0.005'))
slow_ranks = list(range(I - max(1, int(np.ceil(SLOW_RATIO * I))), I))

# SCENARIO=2/3 随机时延参数（与异步程序保持一致）
# SCENARIO=2: 通畅网络——全程 NORMAL_SLEEP + 高斯抖动
# SCENARIO=3: 固定基础时延 + 对数正态抖动——异构节点计算能力 + 随机网络抖动
CONGESTION_JITTER_STD = float(os.environ.get('CONGESTION_JITTER_STD', '0.003'))
LOGNORMAL_MEAN  = float(os.environ.get('LOGNORMAL_MEAN',  '-4.8'))
LOGNORMAL_SIGMA = float(os.environ.get('LOGNORMAL_SIGMA', '0.9'))


# ================= 3. 本地子问题求解器 =================
def compute_x(k, rho, gamma_prox, x_prev_local, hat_x_ij, hat_x_ji, lam_ij, lam_ji, env):
    model = gp.Model(f"Subproblem_{k}", env=env)

    x_ij = model.addVars(neighbors, lb=0, vtype=GRB.INTEGER, name='x_ij')
    x_ji = model.addVars(neighbors, lb=0, vtype=GRB.INTEGER, name='x_ji')

    c_in = model.addVars(J[k], lb=0, vtype=GRB.INTEGER, name='c_in')
    c_out = model.addVars(J[k], lb=0, vtype=GRB.INTEGER, name='c_out')
    w = model.addVars(J[k], J[k], lb=0, vtype=GRB.INTEGER, name='w')
    c = model.addVars(J[k], J[k], lb=0, vtype=GRB.INTEGER, name='c')
    Q = model.addVars(J[k], lb=0, vtype=GRB.INTEGER, name='Q')
    b = model.addVars(J[k], lb=0, vtype=GRB.INTEGER, name='b')

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

    const_cost = 0.5 * 0.25 * 10000 * 2000 / (2**10)
    k1 = gp.quicksum((x_ij[j] + x_ji[j]) * DKC[k][j] * const_cost for j in neighbors)
    k2 = gp.quicksum(
        2000 / 2**10 * b[m] * QY[sum(J[:k]) + m, 7] +
        Q[m] * QY[sum(J[:k]) + m, 5] * 45 * 24 * 365 * 10 * 0.1
        for m in range(J[k])
    )

    k3_ij = gp.LinExpr()
    k3_ji = gp.LinExpr()
    prox_term = gp.LinExpr()
    for j in neighbors:
        avg_ij = 0.5 * (x_prev_local[k, j] + hat_x_ij[j])
        avg_ji = 0.5 * (x_prev_local[j, k] + hat_x_ji[j])

        k3_ij += lam_ij[j] * (x_ij[j] - avg_ij) + (rho / 2.0) * (x_ij[j] - avg_ij) * (x_ij[j] - avg_ij)
        k3_ji += lam_ji[j] * (x_ji[j] - avg_ji) + (rho / 2.0) * (x_ji[j] - avg_ji) * (x_ji[j] - avg_ji)

        prox_term += (gamma_prox / 2.0) * (x_ij[j] - x_prev_local[k, j]) * (x_ij[j] - x_prev_local[k, j])
        prox_term += (gamma_prox / 2.0) * (x_ji[j] - x_prev_local[j, k]) * (x_ji[j] - x_prev_local[j, k])

    model.setObjective(k1 + k2 + k3_ij + k3_ji + prox_term, GRB.MINIMIZE)
    model.optimize()

    ans_x = np.zeros((I, I))
    if model.status == GRB.OPTIMAL:
        true_obj = k1.getValue() + k2.getValue()
        for j in neighbors:
            ans_x[k, j] = x_ij[j].X
            ans_x[j, k] = x_ji[j].X
        return ans_x, true_obj

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
    else:
        time.sleep(NORMAL_SLEEP)


# ================= 4. 主循环（同步） =================
if __name__ == '__main__':
    local_k = 0
    status = MPI.Status()
    wall_start = timer()

    # 仅维护本节点局部矩阵；邻居侧仅缓存与本节点相关的边变量
    x_local = np.zeros((I, I))
    edge_cache_rank_to_nb = {j: 0.0 for j in neighbors}  # 邻居 j 对 x[rank,j] 的副本
    edge_cache_nb_to_rank = {j: 0.0 for j in neighbors}  # 邻居 j 对 x[j,rank] 的副本

    lam_ij = {j: 0.0 for j in neighbors}
    lam_ji = {j: 0.0 for j in neighbors}

    z_prev_ij = {j: 0.0 for j in neighbors}
    z_prev_ji = {j: 0.0 for j in neighbors}

    err_pri = np.zeros(max_iter + 5)
    err_dual = np.zeros(max_iter + 5)
    obj_local_hist = np.zeros(max_iter + 5)
    obj_global_hist = np.zeros(max_iter + 5)
    pri_global_hist = np.zeros(max_iter + 5)
    dual_global_hist = np.zeros(max_iter + 5)
    iter_wall_hist = np.zeros(max_iter + 5)
    eps_pri_hist = np.zeros(max_iter + 5)
    eps_dual_hist = np.zeros(max_iter + 5)
    time_comm = 0.0
    time_comp = 0.0
    msg_send_cnt = 0
    msg_recv_cnt = 0
    bytes_send = 0
    bytes_recv = 0
    final_phys_obj = 0.0
    stable_cnt = 0
    in_fixed_phase = (rho_strategy == "fixed")
    hybrid_switched = False
    rho_freeze_value = rho_fixed_value

    while local_k < max_iter:
        # ---------- Step 1: 同步广播 ----------
        tic = timer()
        send_reqs = []
        for dest in neighbors:
            # 边消息：仅发送与目标邻居 dest 相关的两条边变量
            msg = {
                "t": local_k,
                "x_sender_to_dest": float(x_local[rank, dest]),
                "x_dest_to_sender": float(x_local[dest, rank]),
            }
            send_reqs.append(comm.isend(msg, dest=dest, tag=local_k))
            msg_send_cnt += 1
            bytes_send += len(pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL))

        for src in neighbors:
            data = comm.recv(source=src, tag=local_k, status=status)
            msg_recv_cnt += 1
            bytes_recv += len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
            t_recv = int(data.get("t", -1))
            if t_recv != local_k:
                raise RuntimeError(
                    f"[Rank {rank}] 收到错轮次消息: from={src}, t={t_recv}, expected={local_k}"
                )
            edge_cache_nb_to_rank[src] = float(data["x_sender_to_dest"])
            edge_cache_rank_to_nb[src] = float(data["x_dest_to_sender"])

        MPI.Request.Waitall(send_reqs)
        toc = timer()
        time_comm += toc - tic

        # ---------- Step 2: 组装邻居副本 ----------
        hat_x_ij = {}
        hat_x_ji = {}
        for j in neighbors:
            hat_x_ij[j] = edge_cache_rank_to_nb[j]
            hat_x_ji[j] = edge_cache_nb_to_rank[j]

        # ---------- Step 3: 本地优化 ----------
        
        tic = timer()
        workload(local_k)
        gamma_prox = para * rho # 近端项系数

        x_prev_local = copy.deepcopy(x_local)
        try:
            x_new, phys_obj = compute_x(rank, rho, gamma_prox, x_prev_local, hat_x_ij, hat_x_ji, lam_ij, lam_ji, global_env)
        except Exception as e:
            print(f"[Rank {rank}] 子问题求解失败: {e}", flush=True)
            comm.Abort(1)
        obj_local_hist[local_k] = phys_obj
        final_phys_obj = phys_obj

        # ---------- Step 4: 对偶更新 + 残差 ----------
        pri_sq = 0.0
        dual_sq = 0.0
        norm_x_sq = 0.0
        norm_z_sq = 0.0
        norm_y_sq = 0.0

        for j in neighbors:
            r_ij = x_new[rank, j] - hat_x_ij[j]
            r_ji = x_new[j, rank] - hat_x_ji[j]
            pri_sq += r_ij**2 + r_ji**2

            norm_x_sq += x_new[rank, j]**2 + x_new[j, rank]**2
            norm_z_sq += hat_x_ij[j]**2 + hat_x_ji[j]**2

            # 过度松弛：用前瞻量更新乘子，提升固定 rho 阶段收敛速度
            # alpha_now = alpha_relax_fixed if in_fixed_phase else alpha_relax_adaptive
            alpha_now=1
            relaxed_x_ij = alpha_now * x_new[rank, j] + (1 - alpha_now) * hat_x_ij[j]
            relaxed_x_ji = alpha_now * x_new[j, rank] + (1 - alpha_now) * hat_x_ji[j]
            lam_ij[j] += (rho / 2.0) * (relaxed_x_ij - hat_x_ij[j])
            lam_ji[j] += (rho / 2.0) * (relaxed_x_ji - hat_x_ji[j])

            norm_y_sq += lam_ij[j]**2 + lam_ji[j]**2

            z_curr_ij = 0.5 * (x_new[rank, j] + hat_x_ij[j])
            z_curr_ji = 0.5 * (x_new[j, rank] + hat_x_ji[j])

            dual_sq += (rho * (z_curr_ij - z_prev_ij[j]))**2
            dual_sq += (rho * (z_curr_ji - z_prev_ji[j]))**2

            z_prev_ij[j] = z_curr_ij
            z_prev_ji[j] = z_curr_ji

        err_pri[local_k] = np.sqrt(pri_sq)
        err_dual[local_k] = np.sqrt(dual_sq)
        
        # 动态阈值计算 (ADMM 标准停止准则)
        # eps_pri = sqrt(n)*eps_abs + eps_rel * max(||x||, ||z||)
        # eps_dual = sqrt(n)*eps_abs + eps_rel * ||y||
        # 参与范数计算的变量总数 n = 2 * len(neighbors) (因为有 x_ij 和 x_ji)
        n_vars = 2 * len(neighbors)
        eps_pri_cur = np.sqrt(n_vars) * conv_eps_abs + conv_eps_rel * max(np.sqrt(norm_x_sq), np.sqrt(norm_z_sq))
        eps_dual_cur = np.sqrt(n_vars) * conv_eps_abs + conv_eps_rel * np.sqrt(norm_y_sq)
        eps_pri_hist[local_k] = eps_pri_cur
        eps_dual_hist[local_k] = eps_dual_cur

        x_local = copy.deepcopy(x_new)

        toc = timer()
        time_comp += toc - tic

        # ---------- Step 5: 全局收敛判据（同步） ----------
        local_converged = int((err_pri[local_k] <= eps_pri_cur) and (err_dual[local_k] <= eps_dual_cur))

        # Allgather：全局同步通信，计入 time_comm
        tic = timer()
        send_buf = np.array([local_converged, phys_obj, err_pri[local_k], err_dual[local_k]], dtype=np.float64)
        recv_buf = np.zeros(4 * size, dtype=np.float64)
        comm.Allgather(send_buf, recv_buf)
        toc = timer()
        time_comm += toc - tic

        iter_wall_hist[local_k] = timer() - wall_start
        recv_arr = recv_buf.reshape((size, 4))

        all_converged = int(np.min(recv_arr[:, 0]))
        global_obj = np.sum(recv_arr[:, 1])
        global_pri = np.max(recv_arr[:, 2])
        global_dual = np.max(recv_arr[:, 3])

        if all_converged == 1:
            stable_cnt += 1
        else:
            stable_cnt = 0

        # ---------- Step 6: 自适应 rho（残差平衡） ----------
        obj_global_hist[local_k] = global_obj
        pri_global_hist[local_k] = global_pri
        dual_global_hist[local_k] = global_dual

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
                    rho_freeze_value =  rho_fixed_value
                rho = rho_freeze_value
                in_fixed_phase = True
            else:
                in_fixed_phase = False
                if global_pri > mu * global_dual:
                    rho = min(rho * tau_inc, rho_max)
                elif global_dual > mu * global_pri:
                    rho = max(rho / tau_dec, rho_min)
        else:
            in_fixed_phase = False
            if global_pri > mu * global_dual:
                rho = min(rho * tau_inc, rho_max)
            elif global_dual > mu * global_pri:
                rho = max(rho / tau_dec, rho_min)
        if verbose_iter_log and rank == 0:
            print(
                f"Rank {rank} | Iter {local_k+1:03d} | Obj {phys_obj:.2f} | "
                f"r {err_pri[local_k]:.4e} | s {err_dual[local_k]:.4e} | "
                f"gobj {global_obj:.2f} | rho {rho:.4f} | "
                f"stable {stable_cnt}/{conv_stable_need}"
            )

        local_k += 1
        if all_converged == 1 and stable_cnt >= conv_stable_need:
            break

    # ================= 5. 汇总 =================
    valid_pri = err_pri[:max(local_k, 1)]
    valid_dual = err_dual[:max(local_k, 1)]
    valid_obj_local = obj_local_hist[:max(local_k, 1)]
    valid_iter_wall = iter_wall_hist[:max(local_k, 1)]
    valid_eps_pri = eps_pri_hist[:max(local_k, 1)]
    valid_eps_dual = eps_dual_hist[:max(local_k, 1)]

    all_pri = comm.gather(valid_pri, root=0)
    all_dual = comm.gather(valid_dual, root=0)
    all_eps_pri = comm.gather(valid_eps_pri.tolist(), root=0)
    all_eps_dual = comm.gather(valid_eps_dual.tolist(), root=0)
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
    wall_total = timer() - wall_start
    all_wall = comm.gather(wall_total, root=0)
    valid_obj_global = obj_global_hist[:max(local_k, 1)]
    valid_pri_global = pri_global_hist[:max(local_k, 1)]
    valid_dual_global = dual_global_hist[:max(local_k, 1)]

    if rank == 0:
        total_final_obj = float(np.sum(all_final_obj))
        avg_iter = float(np.mean(all_iters))
        wall_clock_total = float(np.max(all_wall))
        node_avg_runtime = float(np.mean(all_wall))
        avg_comp_time = float(np.mean(all_time_comp))
        avg_comm_time = float(np.mean(all_time_comm))
        comm_ratio = avg_comm_time / (avg_comp_time + avg_comm_time) if (avg_comp_time + avg_comm_time) > 0 else float('nan')
        abs_diff = float('nan')
        rel_diff = float('nan')

        if centralized_base is not None:
            abs_diff = abs(total_final_obj - centralized_base)
            rel_diff = abs_diff / max(abs(centralized_base), 1.0)

        print("\n>>> 同步核心指标（按对比要求）")
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

        print("\n>>> 开始保存迭代轨迹...")
        try:
            trace_len = len(valid_obj_global)
            wall_time_global = np.zeros(trace_len)
            eps_x_global = np.zeros(trace_len)
            eps_u_global = np.zeros(trace_len)
            for i in range(trace_len):
                w_v = [all_iter_wall[r][i] if i < len(all_iter_wall[r]) else all_iter_wall[r][-1] for r in range(size)]
                ep_v = [all_eps_pri[r][i] if i < len(all_eps_pri[r]) else all_eps_pri[r][-1] for r in range(size)]
                ed_v = [all_eps_dual[r][i] if i < len(all_eps_dual[r]) else all_eps_dual[r][-1] for r in range(size)]
                wall_time_global[i] = np.max(w_v)
                eps_x_global[i] = np.max(ep_v)
                eps_u_global[i] = np.max(ed_v)

            df_trace = pd.DataFrame({
                'iter': np.arange(1, trace_len + 1),
                'wall_time_global': wall_time_global,
                'dx_global': valid_pri_global,
                'du_global': valid_dual_global,
                'eps_x_global': eps_x_global,
                'eps_u_global': eps_u_global,
                'obj_global': valid_obj_global,
            })
            sync_trace_path = os.path.join(base_dir, 'admm_trace_sync.csv')
            df_trace.to_csv(sync_trace_path, index=False, encoding='utf-8-sig')
            print(f">>> 同步全局收敛轨迹已保存至: {sync_trace_path}")
        except Exception as e:
            print(f"保存失败: {e}")

    print(f"[Rank {rank}] 通信耗时: {time_comm:.3f}s | 计算耗时: {time_comp:.3f}s")

    comm.Barrier()
    MPI.Finalize()
