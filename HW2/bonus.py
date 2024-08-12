from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import datasets
import numpy as np


iris = datasets.load_iris()

X = iris.data

X = np.append(np.array([np.arange(X.shape[0])]).transpose(), X, 1)

inertia = [0 for _ in range(10)]

for k in range(1, 11):
    model = KMeans(n_clusters=k, init="k-means++", random_state=0, n_init='auto')
    centers = model.fit(X).cluster_centers_

    distances = np.array([float("inf") for _ in range(X.shape[0])])
    for center in centers:
        difference = X - center

        # for each point, calculate its distance from the current center
        dist_from_center = np.sum(difference ** 2, axis = 1)

        # apply minimum row by row
        distances = np.minimum(distances, dist_from_center)  

    inertia[k-1] = np.sum(distances)

ax = plt.figure().gca()

ax.plot([i for i in range(1, 11)], inertia)
ax.set_ylabel("inertia")
ax.set_xlabel("numbrt of cluster")
ax.add_patch(patches.Ellipse(xy=(2.5, 5e4), width=0.8, height=60000, 
                             angle=0.001, linewidth=1, linestyle="--", fill=False))

ax.text(2.7, 65000, "elbow point (k in [2,3])")

plt.savefig("elbow.png")