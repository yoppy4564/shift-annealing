import os
import datetime
import numpy as np
import neal
from pyqubo import Array, Constraint
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# パラメータとデータの設定
num_workers = 4  # 作業員の数 A
num_days = 30  # スケジューリング対象日の数 D
num_terms = 3  # 1日のタームの数 T
num_reads = 10 #試行回数

S = np.ones((num_days, num_terms), dtype=int)  # 必要なブース数
r = np.random.randint(2, size=(num_workers, num_days, num_terms))  # 作業員のシフトに入れるか否か
R = np.ones((num_workers), dtype=int) # 作業員の希望割り当てシフト数
g = np.random.randint(num_workers)# ランダム生成グループ数
G = np.random.randint(2, size=(g, num_workers))  # 作業員が同じグループに所属しているかどうか

# 変数の設定
X = Array.create('X', shape=(num_workers, num_days, num_terms), vartype='BINARY')

# 目的関数と制約の定義
H_terms = sum((sum((X[i, d, t] for i in range(num_workers))) - S[d, t])**2 for d in range(num_days) for t in range(num_terms))
H_workers = sum((sum((X[i, d, t]*r[i, d, t] for d in range(num_days) for t in range(num_terms))) - R[i])**2 for i in range(num_workers))

H_shifts = Constraint(sum((X[i, d, t]*(1 - r[i, d, t]) for i in range(num_workers) for d in range(num_days) for t in range(num_terms))), label='shifts')
H_groups = Constraint(sum((X[i, d, t] - X[j, d, t])**2 for i in range(num_workers) for j in range(i+1, num_workers) for d in range(num_days) for t in range(num_terms) if G[i, j]), label='groups')

# 全体のハミルトニアン
H = H_terms + H_workers + H_shifts + H_groups

# モデルのコンパイル
model = H.compile()

# QUBO形式への変換
qubo, offset = model.to_qubo()

# このQUBO問題を解くためのソルバを使用します（例: D-Wave
sampler = neal.SimulatedAnnealingSampler()
result = sampler.sample_qubo(qubo, num_reads=num_reads)

cmap = mcolors.ListedColormap(['black', 'red'])


for r in result.data(['sample']):
    data = r.sample
    array_data = np.empty((num_workers, num_days, num_terms))

    # Populate the array with the data from the dictionary
    for key, value in data.items():
        indices = [int(i) for i in key[2:-1].split("][")]
        array_data[indices[0], indices[1], indices[2]] = value

    # Create 2D table for each of the 3rd dimension 'k'
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = f"./images_{timestamp}"
    os.makedirs(directory, exist_ok=True) 

    for k in range(num_terms):
        fig, ax = plt.subplots()
        for i in range(num_workers):
            for j in range(num_days):
                cell_data = array_data[i, j, k]
                cell_text = str(k) if cell_data == 1 else ""
                cell = ax.table(cellText=[[cell_text]],
                                cellLoc='center',
                                bbox=[j/num_days, (num_workers-i-1)/num_workers, 1/num_days, 1/num_workers])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    plt.savefig(f"{directory}/table_{k}.svg" , format='svg')  # Save the image in the new directory
