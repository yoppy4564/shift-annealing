import matplotlib.pyplot as plt

# データの準備
data = [['Text1', 'Text2'], ['Text3', 'Text4']]

# カラーマップの設定
colors = [['red', 'blue'], ['green', 'purple']]

# テーブルの作成
fig, ax = plt.subplots()
table = ax.table(cellText=data, cellColours=colors, loc='center')

# セルのスタイル設定
for i in range(len(data)):
    for j in range(len(data[i])):
        cell = table.get_celld()[(i, j)]
        cell.set_text_props(weight='bold')

# 表示
plt.axis('off')
plt.show()
