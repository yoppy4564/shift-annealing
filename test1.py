import os
import datetime
import numpy as np
import neal
from pyqubo import Array, Constraint
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap


# パラメータとデータの設定
num_workers = 4  # 作業員の数 A
num_days = 30  # スケジューリング対象日の数 D
num_terms = 2  # 1日のタームの数 T
num_reads = 10 #試行回数

def create_model():
    S = np.ones((num_days, num_terms), dtype=int)  # 必要なブース数
    r = np.random.randint(2, size=(num_workers, num_days, num_terms), dtype=int)  # 作業員がシフトに入れるか否か
    R = np.ones((num_workers), dtype=int) * 7  # 作業員の希望割り当てシフト数
    g = np.random.randint(num_workers)# ランダム生成グループ数
    G = np.random.randint(2, size=(g, num_workers))  # 作業員が同じグループに所属しているかどうか

    # 変数の設定
    X = Array.create('X', shape=(num_workers, num_days, num_terms), vartype='BINARY')

    # 目的関数と制約の定義
    H_terms = sum((
                sum(
                    (X[i, d, t] for i in range(num_workers))) - S[d, t])**2 for d in range(num_days) for t in range(num_terms))
    H_workers = sum((sum((X[i, d, t]*r[i, d, t] for d in range(num_days) for t in range(num_terms))) - R[i])**2 for i in range(num_workers))

    H_shifts = Constraint(
        sum(
            (X[i, d, t]*(1 - r[i, d, t]) 
             for i in range(num_workers) for d in range(num_days) for t in range(num_terms)
            )
            ), 
            label='shifts'
    )

    H_groups = Constraint(
        sum(
            sum(
                (G[j, i_g] for i_g in range(num_workers))
            ) - sum(
                (X[i_xg1, d, t]*G[j, i_xg1] for i_xg1 in range(num_workers))
            ) * sum(
                (X[i_xg2, d, t] * G[j, i_xg2] for i_xg2 in range(num_workers))
            ) 
            for j in range(g) 
            for d in range(num_days) 
            for t in range(num_terms)
            ), 
            label='groups'
    )

    H_one_shift_per_day = Constraint(
        sum(
            (sum(X[i, d, t] for t in range(num_terms)) 
             - 1)**2 # 各作業員は1日に一つのタームしか勤務できない
            for i in range(num_workers) for d in range(num_days)
        ), 
        label='one_shift_per_day'
    )


    # 全体のハミルトニアン
    H = H_terms + H_workers + H_shifts + H_groups + H_one_shift_per_day
    #H = H_terms + H_shifts + H_groups + H_one_shift_per_day

    # モデルのコンパイル
    model = H.compile()

    # QUBO形式への変換
    qubo, offset = model.to_qubo(feed_dict={"shift":1.0, "groups":1.0, "one_shift_per_day":1.0})
    return qubo, offset, r

def exe(qubo):
    sampler = neal.SimulatedAnnealingSampler()
    result = sampler.sample_qubo(qubo, num_reads=num_reads)
    return result


def plot(result, r):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = f"./images_{timestamp}"
    os.makedirs(directory, exist_ok=True) 

    for idx, sa in enumerate(result.data(['sample']), start=1):
        sample = sa.sample
        X = np.empty((num_workers, num_days, num_terms), dtype=int)

        # Populate the array with the data from the dictionary
        for key, value in sample.items():
            indices = [int(i) for i in key[2:-1].split("][")]
            X[indices[0], indices[1], indices[2]] = value
        
        # 各セルの色を決定する配列を作成
        colors = np.empty((num_workers, num_days), dtype=str)
        for i in range(num_workers):
            for j in range(num_days):
                if np.sum(X[i, j, :]) > 1:  # 同じ日に複数のタームが割り当てられている場合
                    colors[i, j] = 'red'
                elif (X[i, j, :] == 1).any() and (r[i, j, :] == 0).any():  # Xが1でrが0の場合
                    colors[i, j] = 'yellow'
                else:
                    colors[i, j] = 'white'

        # テーブルの値を決定する配列を作成
        values = np.empty((num_workers, num_days), dtype=str)
        for i in range(num_workers):
            for j in range(num_days):
                assigned_terms = np.where(X[i, j, :] == 1)[0]
                if len(assigned_terms) > 0:
                    values[i, j] = ','.join(str(term+1) for term in assigned_terms)  # タームの番号は1から始まると仮定
                else:
                    values[i, j] = ''

        # テーブルを描画
        fig, ax = plt.subplots()
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=values, cellColours=colors, cellLoc='center', loc='center')

        plt.savefig(f"{directory}/table_{idx}.svg" , format='svg')  # Save the image in the new directory

def plot_combined_schedule(r, X):
    fig, ax = plt.subplots(figsize=(num_days, num_workers))
    ax.set_title('Combined schedule')
    ax.set_xlabel("Days")
    ax.set_ylabel("Workers")
    ax.set_xticks(range(num_days * num_terms))
    ax.set_yticks(range(num_workers))
    ax.set_xticklabels([(day % num_days + 1) for day in range(num_days * num_terms)])
    ax.set_yticklabels(range(1, num_workers+1))

    for i in range(num_workers):
        for j in range(num_days * num_terms):
            cell_text = str(r[i, j])
            ax.text(j, i, cell_text, va='center', ha='center', color='white')
    
    ax.imshow(X, cmap='bwr', alpha=0.5)
    plt.gca().invert_yaxis()

if __name__ == "__main__":
    qubo, offset, r= create_model()
    result = exe(qubo)
    plot(result, r)
