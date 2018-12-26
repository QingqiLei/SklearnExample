import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification

X, Y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, n_classes=3)
# X: tuple, Y label
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
print(X[0:1])
plt.show()

from sklearn import neighbors

clf = neighbors.KNeighborsClassifier(n_neighbors=15, weights='distance')
clf.fit(X, Y)
'''
n_neighbors: default 5 ,不适用于限定半径最近邻法

radius： 不适用于KNN， default 1

weights: 'uniform', 'distance'
algorithm: 'brute','kd_tree','ball_tree','auto'
leaf_size： default 30, 和样本正相关
metric： 距离度量， 默认欧式距离
p： 默认2，表示欧式距离
n_jobs ： default -1, 
outlier_label: 异常点类别选择，不适用KNN

'''

from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# 确认训练集的边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # first column
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # second column
print(x_min, x_max, y_min, y_max)
# 生成随机数据来做测试集，然后作预测
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),  # first and second column
                     np.arange(y_min, y_max, 0.05))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

print(xx.shape)
print(yy.shape)
print(Z.shape)
# 画出测试集数据
Z = Z.reshape(xx.shape)

fig = plt.figure()
ax = fig.add_subplot(222)
ax.scatter(xx, yy, Z, Z)
plt.title('add_subplot')
plt.show()

plt.scatter(xx, yy, Z, Z)
plt.title('plt.scatter()')
plt.show()

# 通过数千数万个点把画布涂色
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.title('color')
plt.show()

# 也画出所有的训练集数据
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = 15, weights = 'distance')")
plt.show()
