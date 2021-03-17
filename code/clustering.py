
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
from rank_list_fusion import fusion
from evaluate import hprec, precision_at_k, rprec, ap, rank_list_from_distances_matrix, distances_matrix_from_rank_list, dataset_infos
import s_curves

print("--Loading dataset--")
# Loading dataset
# id, x, y = oxquarry.parse()
id, x_lemma, x_token, y = brunet.parse()

# Select data
start = 0
lim = -1
X = x_token[start:]
X2 = x_lemma[start:]
Y = y[start:]

print(f"#Texts #Authors Mean_length #Links")
print(*dataset_infos(X, Y))


print("--Linking--")
distances_matrix_A, rank_list_A, mesures_A = experiment(
    X, Y, 6, 500, True, distances.manhattan)
print(f"AP RPrec HPrec", *mesures_A)

distances_matrix_B, rank_list_B, mesures_B = experiment(
    X, Y, 0, 500, True, distances.manhattan)
print(f"AP RPrec HPrec", *mesures_B)

distances_matrix_C, rank_list_C, mesures_C = experiment(
    X2, Y, 0, 500, True, distances.manhattan)
print(f"AP RPrec HPrec", *mesures_C)

rank_lists = [rank_list_A, rank_list_B, rank_list_C]
rank_list_overall = fusion(rank_lists, s_curve=s_curves.sigmoid, args={"alpha":10e-10})
distances_matrix_overall = distances_matrix_from_rank_list(rank_list_overall)

mesures_overall = ap(rank_list_overall, Y), rprec(
    rank_list_overall, Y), hprec(rank_list_overall, Y)
print(f"AP RPrec HPrec", *mesures_overall)

print("--Linking analysis")

distance_threshold = 0.36
pos = len([d for indices, d in rank_list_overall if d < distance_threshold])
print(f"distance_threshold", distance_threshold, pos)

original = np.array(list(dict(rank_list_overall).values()))
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
ac.fit(distances_matrix_overall)

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
