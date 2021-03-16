
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
X = x_token[0:200]
Y = y[0:200]


print("--Linking--")
distances_matrix, rank_list, mesures = experiment(
    X, Y, 6, 500, True, distances.manhattan)
print(f"AP RPrec HPrec", *mesures)

distances_matrix_B, rank_list_B, mesures_B = experiment(
    X, Y, 0, 500, True, distances.manhattan)
print(f"AP RPrec HPrec", *mesures_B)

print("--Linking analysis")

pos = round(len(rank_list) ** 0.46)
distance_threshold = rank_list[pos][-1]
distance_threshold *= 1.28
print(f"distance_threshold", distance_threshold, pos)


original = np.array(list(dict(rank_list).values()))
plt.figure()
plt.plot(range(len(original)), original)
plt.vlines([pos], ymin=min(original), ymax=max(original), colors="r")
plt.savefig("distance_over_rank.png")

print("--Clustering--")


args = {
    "n_clusters": None,
    "affinity": "precomputed",
    "linkage": "average",
    "distance_threshold": distance_threshold
}
ac = AgglomerativeClustering(**args)
ac.fit(distances_matrix)

print("--Clustering Evaluation--")

# print y and cluster
# for a, b in sorted(zip(Y, ac.labels_)):
#     print(a, b)

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

print("adjusted_mutual_info_score", adjusted_mutual_info_score(Y, ac.labels_))
