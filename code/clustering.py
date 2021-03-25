"""Clustering module."""

import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import numpy as np

import distances
from corpus import oxquarry, brunet, st_jean, pan16
from rank_list_fusion import compute_multiple_links
from misc import dataset_infos, distances_matrix_from_rank_list
from evaluate import evaluate_linking, evaluate_clustering
import s_curves


def clustering(rank_list):
    distances_matrix = distances_matrix_from_rank_list(rank_list)
    rank_list_threshold = int(len(rank_list) / 3)
    distance_thresholds = [i[-1] for i in rank_list[0:rank_list_threshold]]
    silhouette_scores = []
    best_score = -np.inf
    best_ac = None

    for distance_threshold in distance_thresholds:
        args = {
            "n_clusters": None,
            "affinity": "precomputed",
            "linkage": "average",
            "distance_threshold": distance_threshold
        }
        ac = AgglomerativeClustering(**args)
        ac.fit(distances_matrix)

        try:
            score = silhouette_score(distances_matrix, ac.labels_)
        except ValueError:
            score = 0
        silhouette_scores.append(score)
        if score > best_score:
            best_score = score
            best_ac = ac

    return best_ac, silhouette_scores


if __name__ == '__main__':
    print("--Loading dataset--")
    _, X, Y = oxquarry.parse()
    # _, _, X, Y = brunet.parse()
    # _, _, X, Y = pan16.parse_train()[0]


    print("#Texts #Authors Mean_length #Links")
    print(*dataset_infos(X, Y))

    print("--Linking--")
    experiments = [
        [X, 0, 500, False, 0.1, distances.clark],
        [X, 0, 500, False, 0.1, distances.tanimoto],
        [X, 0, 500, True, 0.1, distances.manhattan],
        [X, 5, 500, False, 0.1, distances.tanimoto],
    ]
    s_curve = s_curves.sigmoid_reciprocal

    rank_list_overall, rank_lists = compute_multiple_links(
        experiments, s_curve)

    print("--Linking evaluation--")
    print("AP RPrec HPrec (Used for overall)")
    for rank_list in rank_lists:
        mesures = evaluate_linking(rank_list, Y)
        print(*mesures)
    print("AP RPrec HPrec (Overall)")
    mesures = evaluate_linking(rank_list_overall, Y)
    print(*mesures)

    print("--Clustering--")
    ac, silhouette_scores = clustering(rank_list_overall)

    plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(range(len(silhouette_scores)), silhouette_scores)
    plt.tight_layout()
    plt.savefig("silhouette_score.png")

    d_threshold = ac.get_params()["distance_threshold"]
    pos = len([d for indices, d in rank_list_overall if d < d_threshold])
    print("distance_threshold", d_threshold, pos)

    original = np.array(list(dict(rank_list_overall).values()))
    plt.figure(figsize=(4, 3), dpi=200)
    plt.vlines([pos], ymin=min(original), ymax=max(original), colors="r")
    plt.hlines([d_threshold], xmin=0, xmax=len(original), colors="r")
    plt.plot(range(len(original)), original)
    plt.tight_layout()
    plt.savefig("distance_over_rank.png")

    print("--Clustering Evaluation--")
    b3_precision, b3_recall, b3_fscore, mis = evaluate_clustering(
        Y, ac.labels_)
    print("bcubed.precision", b3_precision)
    print("bcubed.recall", b3_recall)
    print("bcubed.fscore", b3_fscore)
    print("adjusted_mutual_info_score", mis)
