
from collections import Counter

import bcubed
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
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
X = x_token[start:]
X2 = x_lemma[start:]
Y = y[start:]

print(f"#Texts #Authors Mean_length #Links")
print(*dataset_infos(X, Y))

print("--Linking--")
distances_matrix_A, rank_list_A, mfw_A, mesures_A = experiment(
    X, Y, 5, 500, True, distances.manhattan)
print(f"AP RPrec HPrec", *mesures_A)

distances_matrix_B, rank_list_B, mfw_B, mesures_B = experiment(
    X, Y, 0, 500, True, distances.manhattan)
print(f"AP RPrec HPrec", *mesures_B)

distances_matrix_C, rank_list_C, mfw_C, mesures_C = experiment(
    X2, Y, 5, 500, True, distances.manhattan)
print(f"AP RPrec HPrec", *mesures_C)

distances_matrix_D, rank_list_D, mfw_D, mesures_D = experiment(
    X2, Y, 0, 500, True, distances.manhattan)
print(f"AP RPrec HPrec", *mesures_D)

rank_lists = [rank_list_A, rank_list_B, rank_list_C, rank_list_D]
rank_list_overall = fusion(rank_lists, s_curve=s_curves.sigmoid)
distances_matrix_overall = distances_matrix_from_rank_list(rank_list_overall)

mesures_overall = ap(rank_list_overall, Y), rprec(
    rank_list_overall, Y), hprec(rank_list_overall, Y)
print(f"AP RPrec HPrec", *mesures_overall)

print("--Clustering--")

plt.figure(figsize=(4, 3), dpi=200)
l = len(rank_list_overall)
distance_thresholds = [i[-1] for i in rank_list_overall[0:int(l/3)]]
scores = []
best_score= -np.inf
best_ac = None
for distance_threshold in distance_thresholds:
    args = {
        "n_clusters": None,
        "affinity": "precomputed",
        "linkage": "average",
        "distance_threshold": distance_threshold
    }
    ac = AgglomerativeClustering(**args)
    ac.fit(distances_matrix_overall)
    try:
        score = silhouette_score(distances_matrix_overall, ac.labels_)
    except ValueError:
        score = 0
    scores.append(score)
    if score > best_score:
        best_score = score
        best_ac = ac
plt.plot(range(len(distance_thresholds)), scores)
plt.tight_layout()
plt.savefig("silhouette_score.png")

ac = best_ac

distance_threshold = ac.get_params()["distance_threshold"]
pos = len([d for indices, d in rank_list_overall if d < distance_threshold])
print(f"distance_threshold", distance_threshold, pos)

original = np.array(list(dict(rank_list_overall).values()))
plt.figure(figsize=(4, 3), dpi=200)
plt.vlines([pos], ymin=min(original), ymax=max(original), colors="r")
plt.hlines([distance_threshold], xmin=0, xmax=len(original), colors="r")
plt.plot(range(len(original)), original)
plt.tight_layout()
plt.savefig("distance_over_rank.png")


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
