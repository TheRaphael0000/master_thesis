
from collections import Counter

import bcubed
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_mutual_info_score

import distances
from corpus import oxquarry, brunet, st_jean
from linking import experiment, compute_links

# Loading dataset
# id, x, y = oxquarry.parse()
id, x_lemma, x_token, y = brunet.parse()

# Select data
X = x_lemma
Y = y

distances_matrix, rank_list, mesures = experiment(
    X, Y, 5, 500, True, distances.manhattan)

pos = round(len(rank_list) ** 0.46)

distance_threshold = rank_list[pos][-1]
distance_threshold *= 1.2

A = range(len(rank_list))
B = dict(rank_list).values()
plt.plot(A, B)
plt.vlines([pos], ymin=min(B), ymax=max(B), colors="r")
plt.savefig("distance_over_rank.png")

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

print()

print(f"distance_threshold")
print(distance_threshold)
print(pos)

print(f"AP RPrec HPrec")
print(mesures)

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
