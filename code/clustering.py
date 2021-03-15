
from collections import Counter

import bcubed
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_mutual_info_score

from scipy import signal
import numpy as np

import distances
from corpus import oxquarry, brunet, st_jean
from linking import experiment, compute_links

print("--Loading dataset--")
# Loading dataset
# id, x, y = oxquarry.parse()
id, x_lemma, x_token, y = st_jean.parse()

# Select data
X = x_lemma[0:50]
Y = y[0:50]


print("--Linking--")
distances_matrix, rank_list, mesures = experiment(
    X, Y, 5, 500, True, distances.manhattan)


print("--Clustering--")
pos = round(len(rank_list) ** 0.46)

distance_threshold = rank_list[pos][-1]
distance_threshold *= 1.2

#-- derivation experiments
original = np.array(list(dict(rank_list).values()))

window_size = 2*(30) + 1
win = np.ones(window_size) / window_size
derivative_kernel = np.array([-1, 1])

smoothed = signal.convolve(original, win, mode="valid")
dx = signal.convolve(smoothed, derivative_kernel, mode="valid")
dx_smoothed = signal.convolve(dx, win, mode="valid")

plt.figure()
plt.plot(range(len(original)), original)
plt.vlines([pos], ymin=min(original), ymax=max(original), colors="r")

plt.figure()
plt.plot(range(len(smoothed)), smoothed)

plt.figure()
plt.plot(range(len(dx)), dx)

plt.figure()
plt.plot(range(len(dx_smoothed)), dx_smoothed)

# plt.show()
#-- derivation experiments end

args = {
    "n_clusters": None,
    "affinity": "precomputed",
    "linkage": "average",
    "distance_threshold": distance_threshold
}

ac = AgglomerativeClustering(**args)

ac.fit(distances_matrix)

for a, b in sorted(zip(Y, ac.labels_)):
    print(a, b)

print("--Evaluation--")

print(f"distance_threshold", distance_threshold, pos)

print(f"AP RPrec HPrec", *mesures)

print("adjusted_mutual_info_score", adjusted_mutual_info_score(Y, ac.labels_))

ldict = {}
cdict = {}
for i, (l, c) in enumerate(zip(Y, ac.labels_)):
    ldict[i] = set([l])
    cdict[i] = set([c])
precision = bcubed.precision(cdict, ldict)
recall = bcubed.recall(cdict, ldict)
fscore = bcubed.fscore(precision, recall)
print("bcubed.precision", precision)
print("bcubed.recall", recall)
print("bcubed.fscore", fscore)
