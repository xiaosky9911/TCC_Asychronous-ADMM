# -*- coding: utf-8 -*-
"""
集中式（非分布式）Gurobi 优化：一次性优化所有区域，计算运行成本。

用法：
python d:\yanjiusheng\ready\Cloud_TCC_Revision\Centralized_cost.py need_data_N4
python d:\yanjiusheng\ready\Cloud_TCC_Revision\Centralized_cost.py need_data_N8
python d:\yanjiusheng\ready\Cloud_TCC_Revision\Centralized_cost.py need_data_N16

$env:CENTRALIZED_BASE="481369566.306250"; 
$env:CENTRALIZED_BASE="1182823039.614968"; 
$env:CENTRALIZED_BASE="2459936083.674638"; 
$env:CENTRALIZED_BASE="4910228836.029585"; 

也支持环境变量：
set DATASET_NAME=need_data_N4
python d:\yanjiusheng\ready\Cloud_TCC_Revision\Centralized_cost.py
"""

import os
import sys
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


def main():
    base_dir = r"d:\yanjiusheng\ready\Cloud_TCC_Revision"

    dataset_options = {
        "need_data_N32": {
            "file": os.path.join(base_dir, "need_data_N32.xlsx"),
            "I": 32,
            "J": [3] * 32,
        },

        "need_data_N16": {
            "file": os.path.join(base_dir, "need_data_N16.xlsx"),
            "I": 16,
            "J": [3] * 16,
        },
        "need_data_N8": {
            "file": os.path.join(base_dir, "need_data_N8.xlsx"),
            "I": 8,
            "J": [3, 3, 3, 3, 3, 3, 3, 3],
        },
        "need_data_N4": {
            "file": os.path.join(base_dir, "need_data_N4.xlsx"),
            "I": 4,
            "J": [3, 3, 2, 2],
        },
    }

    dataset_name = os.environ.get("DATASET_NAME", "need_data_N4")
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]

    if dataset_name not in dataset_options:
        raise ValueError(f"未知数据集: {dataset_name}，可选项: {list(dataset_options.keys())}")

    cfg = dataset_options[dataset_name]
    io = cfg["file"]
    I = cfg["I"]
    J = cfg["J"]

    # 读取数据
    fz = pd.read_excel(io, sheet_name="城市参数")
    qy = fz.values[0:sum(J), 1:9]

    dkq_data = pd.read_excel(io, sheet_name="带宽上限")
    dkq = dkq_data.values[0:I, 1:I + 1]

    dkc_data = pd.read_excel(io, sheet_name="单价系数")
    dkc = dkc_data.values[0:I, 1:I + 1]

    # 每个区域的城市全局行号
    city_rows = []
    start = 0
    for k in range(I):
        city_rows.append(list(range(start, start + J[k])))
        start += J[k]

    # 建模
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model("centralized_cost", env=env)

        # 区域间流量变量
        x = model.addVars(I, I, lb=0, vtype=GRB.CONTINUOUS, name="x")

        # 区域内变量（用字典索引避免二维不规则）
        c1 = {}
        c2 = {}
        Q = {}
        b = {}
        w = {}
        c = {}

        for k in range(I):
            for m in range(J[k]):
                c1[k, m] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"c1_{k}_{m}")
                c2[k, m] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"c2_{k}_{m}")
                Q[k, m] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"Q_{k}_{m}")
                b[k, m] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"b_{k}_{m}")
            for m in range(J[k]):
                for n in range(J[k]):
                    w[k, m, n] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"w_{k}_{m}_{n}")
                    c[k, m, n] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"c_{k}_{m}_{n}")

        # x(i,i)=0
        model.addConstrs((x[i, i] == 0 for i in range(I)), name="x_diag_zero")

        # 约束
        for k in range(I):
            # 入/出区域流量平衡
            model.addConstr(gp.quicksum(x[i, k] for i in range(I)) == gp.quicksum(c1[k, m] for m in range(J[k])),
                            name=f"flow_in_{k}")
            model.addConstr(gp.quicksum(x[k, j] for j in range(I)) == gp.quicksum(c2[k, m] for m in range(J[k])),
                            name=f"flow_out_{k}")

            # 每个城市内部约束
            for m in range(J[k]):
                row = city_rows[k][m]

                model.addConstr(gp.quicksum(w[k, m, n] for n in range(J[k])) == qy[row, 2],
                                name=f"w_row_sum_{k}_{m}")

                model.addConstr(gp.quicksum(c[k, m, n] for n in range(J[k])) + c2[k, m] == qy[row, 1],
                                name=f"c_row_sum_{k}_{m}")

                model.addConstr(
                    qy[row, 1] + qy[row, 2] + qy[row, 3]
                    + gp.quicksum(c[k, n, m] for n in range(J[k]))
                    - gp.quicksum(c[k, m, n] for n in range(J[k]))
                    + c1[k, m] - c2[k, m]
                    + gp.quicksum(w[k, n, m] for n in range(J[k]))
                    - gp.quicksum(w[k, m, n] for n in range(J[k]))
                    == Q[k, m],
                    name=f"balance_Q_{k}_{m}"
                )

                model.addConstr(Q[k, m] <= qy[row, 4], name=f"Q_cap_{k}_{m}")

                model.addConstr(
                    0.25 * 10000 * (
                        gp.quicksum(c[k, n, m] for n in range(J[k]))
                        + gp.quicksum(c[k, m, n] for n in range(J[k]))
                        - 2 * c[k, m, m]
                        + c1[k, m] + c2[k, m]
                    )
                    + 0.3 * 10000 * (
                        gp.quicksum(w[k, n, m] for n in range(J[k]))
                        + gp.quicksum(w[k, m, n] for n in range(J[k]))
                        - 2 * w[k, m, m]
                    )
                    == b[k, m],
                    name=f"b_def_{k}_{m}"
                )

                model.addConstr(b[k, m] <= 2 ** 20 * qy[row, 6], name=f"b_cap_{k}_{m}")

        # 区域间带宽上限
        for i in range(I):
            for j in range(I):
                model.addConstr(0.25 * 10000 * (x[i, j] + x[j, i]) <= 2 ** 20 * dkq[i, j],
                                name=f"bw_{i}_{j}")

        # 目标函数：总运行成本
        const_cost = 0.5 * 0.25 * 10000 * 2000 / (2 ** 10)

        inter_cost = gp.quicksum((x[i, j] + x[j, i]) * dkc[i, j] * const_cost for i in range(I) for j in range(I))
        b_cost = gp.quicksum(
            2000 / (2 ** 10) * b[k, m] * qy[city_rows[k][m], 7]
            for k in range(I)
            for m in range(J[k])
        )
        q_energy_cost = gp.quicksum(
            Q[k, m] * qy[city_rows[k][m], 5] * 45 * 24 * 365 * 10 * 0.1
            for k in range(I)
            for m in range(J[k])
        )

        model.setObjective(inter_cost + b_cost + q_energy_cost, GRB.MINIMIZE)
        model.optimize()

        if model.status != GRB.OPTIMAL:
            raise RuntimeError(f"优化失败，Gurobi 状态码: {model.status}")

        # 输出每个区域的运行成本（按原公式拆分）
        region_costs = []
        for k in range(I):
            k1 = sum((x[k, j].X + x[j, k].X) * dkc[k, j] * const_cost for j in range(I))
            k2 = 0.0
            for m in range(J[k]):
                row = city_rows[k][m]
                k2 += 2000 / (2 ** 10) *b[k, m].X * qy[row, 7]
                k2 += Q[k, m].X * qy[row, 5] * 45 * 24 * 365 * 10 * 0.1
            region_costs.append(k1 + k2)

        transmission_cost = inter_cost.getValue() + b_cost.getValue()
        energy_cost = q_energy_cost.getValue()
        total_cost = sum(region_costs)

        print("=" * 72)
        print(f"集中式优化完成 | 数据集: {dataset_name}")
        print(f"总目标函数值（模型 objective）: {model.ObjVal:.6f}")
        print(f"传输成本（k1 + k2第一部分）: {transmission_cost:.6f}")
        print(f"能耗成本（Q 对应成本）: {energy_cost:.6f}")
        print(f"所有区域运行成本之和（按区域汇总）: {total_cost:.6f}")
        print("各区域运行成本：")
        for k, v in enumerate(region_costs):
            print(f"  区域 {k}: {v:.6f}")
        print("=" * 72)


if __name__ == "__main__":
    main()
