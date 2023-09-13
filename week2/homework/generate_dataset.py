import numpy as np

number_class = 5
number_sample = 1000
number_feature = 5

X = np.random.rand(number_sample, number_feature)
y = np.random.randint(0,number_class, size=number_sample)

# cluster x
# 这样的结果是:
# 类别0的数据集中在原点附近
# 类别1的数据集中在(1,1,...,1)附近
# 类别2的数据集中在(2,2,...,2)附近
for c in range(number_class):
    X[y == c] += c

#one-hot
# y = np.eye(number_class)[y]

# train test split
train_ratio = 0.8
train_size = int(number_sample * train_ratio)
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

# save data
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)