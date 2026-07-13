import os
import subprocess
# 这个脚本用于快速配置环境变量并运行 MPI 任务，适用于 RE_Synchronous.py、RE_Asynchronous.py 等。

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= 1. 基础配置 =================
# 数据集选项: need_data_N4, need_data_N8, need_data_N16

NUM_NODES =8     # 必须与数据集节点数对应 (4, 8, 16)

if NUM_NODES == 4:
    centralized_base = "481369566.306250"
    DATASET = "need_data_N4"
elif NUM_NODES == 8:
    centralized_base = "1182823039.614968"
    DATASET = "need_data_N8"
elif NUM_NODES == 16:
    centralized_base = "2459936083.674638"
    DATASET = "need_data_N16"
elif NUM_NODES == 32:
    centralized_base = "4910228836.029585"
    DATASET = "need_data_N32"

# 运行模式选择: "sync" 或 "async"
MODE = "async"

# 自动根据a模式选择对应的脚本文件
if MODE == "sync":
    SCRIPT_PATH = os.path.join(BASE_DIR, "RE_Synchronous.py")
else:
    SCRIPT_PATH = os.path.join(BASE_DIR, "RE_Asynchronous.py")

# ================= 2. 参数调整 =================
# 通用参数 (同步与异步皆可用)
ENV_VARS = {
    # SCENARIO 选择：
    #   1 = 固定两节点慢（CASE1 对照实验）
    #   2 = 通畅网络 + 高斯抖动（随机时延实验 A）
    #   3 = 固定基础时延 + 对数正态抖动（随机时延实验 B）
    "SCENARIO": "2",
    "MAX_ITER": "3500",       # 算法最大迭代次数
    "RHO_STRATEGY": "fixed",  # rho策略: fixed / adaptive / hybrid / staged_fixed
    "RHO_FIXED": "160",       # 固定 rho 的值
    "PROX_RATIO": "1.5",        # 近端项比例
    "SLOW_RATIO": "0.25",     # 固定掉队者比例（SCENARIO=0/1 使用）
    "SLOW_SLEEP": "0.04",     # 慢状态睡眠时长（所有 SCENARIO 共用）
    "NORMAL_SLEEP": "0.005",  # 快状态睡眠时长（所有 SCENARIO 共用）
    # SCENARIO=2 通畅网络参数
    "CONGESTION_JITTER_STD": "0.003",  # 高斯抖动标准差（s），仅 SCENARIO=2
    # SCENARIO=3 对数正态参数
    # 各 rank 基础时延：[0.004,0.005,0.008,0.006,0.007,0.020,0.005,0.009]
    # 放大系数：        [1.0,  1.0,  1.5,  1.1,  1.2,  2.0,  1.0,  1.6 ]
    # 对数正态抖动均值 e^(-4.8+0.9²/2) ≈ 0.011s，99th 分位 ≈ 0.12s
    "LOGNORMAL_MEAN":  "-4.8",   # 对数正态 μ 参数
    "LOGNORMAL_SIGMA": "1",    # 对数正态 σ 参数
    # 收敛精度参数 (同步与异步均适用)
    "CONV_EPS_ABS": "0.01",   # 绝对收敛阈值 eps_abs
    "CONV_EPS_REL": "0.01",   # 相对收敛阈值 eps_rel
    "CONV_STABLE_NEED": "6",  # 连续满足收敛条件的轮数
    "CENTRALIZED_BASE": centralized_base, # 集中式目标值，用于计算误差
    "BASE_DIR": BASE_DIR,
}

# 异步特有参数 (仅当 MODE == "async" 时注入)
if MODE == "async":
    ENV_VARS.update({
        "MIN_DEP_SET": "6",       # S: 最小依赖邻居数
        "MAX_LAG": "16",          # tau: 最大滞后轮数
    })

# ================= 3. 执行逻辑 =================
def main():
    print("开始配置并运行 MPI 任务...\n")
    
    # 获取当前的系统环境变量副本
    env = os.environ.copy()
    
    # 注入我们自定义的变量
    for key, value in ENV_VARS.items():
        if value is not None:
            env[key] = str(value)
            print(f"  [参数注入] {key} = {value}")
            

    # 构造运行指令
    cmd = [
        "mpiexec", "-np", str(NUM_NODES),
        "python", SCRIPT_PATH, DATASET
    ]
    
    print("\n" + "="*60)
    print(f"执行终端指令: {' '.join(cmd)}")
    print("="*60 + "\n")
    
    # 运行 MPI，并将输出实时显示在终端
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n[中断] 用户提前终止了程序。")
    except Exception as e:
        print(f"\n[错误] 运行失败: {e}")

if __name__ == "__main__":
    main()
