from matplotlib import pyplot as plt
import numpy as np
import random
from collections import defaultdict
random.seed(42)

# create data
def addData(data, num_instance, lowx, lowy, highx, highy):
  for i in range(num_instance):
    x = random.randint(lowx, highx)
    y = random.randint(lowy, highy)
    data.append([x,y])


data = []
# data = [[2,14], [12,16], ..., []]
num_instance = 100
addData(data, num_instance, 0, 0, 20, 20)
addData(data, num_instance, 40, 0, 60, 20)
addData(data, num_instance, 20, 50, 40, 70)

# x and y values in data
x_values = []
y_values = []
for d in data:
    x_values.append(d[0])
    y_values.append(d[1])

# random centers
k = 3
max_iter = 300
centroids = []
for i in range(k):
  random_idx = random.randint(0, len(data)-1)
  centroids.append(data[random_idx])

cent_x = []
cent_y = []
for d in centroids:
    cent_x.append(d[0])
    cent_y.append(d[1])

#plot data and random centers
plt.plot(x_values, y_values, "ro")
plt.plot(cent_x, cent_y, "bo")
plt.show()

print('done')

# find distances
max_iter = 300
cluster_num = None

for iter in range(max_iter):
    cluster = {0: [], 1: [], 2: []}
    for i in range(len(x_values)):
        min_distance = 10000000
        for j in range(len(cent_y)):
            distance = np.sqrt(((x_values[i] - cent_x[j])**2) + ((y_values[i] - cent_y[j])**2))
            if distance < min_distance:
                min_distance = distance
                cluster_num = j
        cluster[cluster_num].append([x_values[i], y_values[i]])

    cent_x = []
    cent_y = []
    assigned_x = []
    assigned_y = []
    for j in range(k):
        assigned_x = [item[0] for item in cluster[j]]
        assigned_y = [item[1] for item in cluster[j]]
        cent_x.append(sum(assigned_x) / len(assigned_x))
        cent_y.append(sum(assigned_y) / len(assigned_y))

plt.figure()
colors = ["red", "green", "blue"]
for j in range(k):
    assigned_x = [item[0] for item in cluster[j]]
    assigned_y = [item[1] for item in cluster[j]]
    plt.plot(assigned_x, assigned_y, "o", color=colors[j])
plt.show()

print('done')