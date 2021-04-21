"""Clustering module."""

import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import distances
import s_curves
from corpus import oxquarry
from corpus import brunet
from corpus import st_jean
from corpus import pan16
from rank_list_fusion import fusion_s_curve_score
from rank_list_fusion import fusion_z_score
from linking import compute_links
from misc import dataset_infos
from misc import distances_matrix_from_rank_list
from misc import rank_list_to_txt
from evaluate import evaluate_linking
from evaluate import evaluate_clustering


def unsupervised_clustering(rank_list, return_scores=False, return_threshold=False):
    distances_matrix = distances_matrix_from_rank_list(rank_list)
    rank_list_threshold = int(len(rank_list) / 1)
    distance_thresholds = [i[-1] for i in rank_list[0:rank_list_threshold]]
    silhouette_scores = []
    best_score = -np.inf
    best_labels = None
    best_distance_threshold = None

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
            print(score)
            best_score = score
            best_labels = ac.labels_
            best_distance_threshold = distance_threshold

    outputs = [best_labels]

    if return_scores:
        outputs += [silhouette_scores]
    if return_threshold:
        outputs += [best_distance_threshold]

    return tuple(outputs)


def clustering_at_every_rank(rank_list):
    distances_matrix = distances_matrix_from_rank_list(rank_list)
    labels_list = []
    distance_thresholds = [i[-1] for i in rank_list]
    for distance_threshold in distance_thresholds:
        args = {
            "n_clusters": None,
            "affinity": "precomputed",
            "linkage": "average",
            "distance_threshold": distance_threshold
        }
        ac = AgglomerativeClustering(**args)
        ac.fit(distances_matrix)
        labels_list.append(ac.labels_)
    return labels_list


def clustering_case():
    # _, X, Y = oxquarry.parse()
    _, _, X, Y = brunet.parse()
    # _, _, _, X, Y = st_jean.parse_A()

    experiments = [
        [X, 0, 500, True, 1e-1, distances.manhattan],
        # [X, 0, 500, False, 1e-1, distances.tanimoto],
        [X, 0, 500, False, 1e-1, distances.clark],
        # [X, 0, 500, False, 1e-1, distances.matusita],
        [X, 0, 500, True, 1e-1, distances.cosine_distance],

        [X, 3, 750, True, 1e-1, distances.manhattan],
        # [X, 3, 750, False, 1e-1, distances.tanimoto],
        # [X, 3, 750, False, 1e-1, distances.clark],
        # [X, 3, 750, False, 1e-1, distances.matusita],
        # [X, 3, 750, False, 1e-1, distances.cosine_distance],
    ]
    s_curve = s_curves.sigmoid_reciprocal()

    print("AP RPrec HPrec")
    rank_lists = [compute_links(*e) for e in experiments]
    for rank_list in rank_lists:
        print(evaluate_linking(rank_list, Y))

    print("Overall")
    rank_list_overall = fusion_z_score(rank_lists)
    print(evaluate_linking(rank_list_overall, Y))

    labels, silhouette_scores, d_threshold = unsupervised_clustering(
        rank_list_overall, return_scores=True, return_threshold=True)

    pos = len([d for indices, d in rank_list_overall if d < d_threshold])
    print("distance_threshold", d_threshold, pos)

    print("bcubed.precision", "bcubed.recall", "bcubed.fscore")
    print(evaluate_clustering(Y, labels))

    labels_list = clustering_at_every_rank(rank_list_overall)
    evaluations = np.array([evaluate_clustering(Y, labels) for labels in labels_list])

    plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(silhouette_scores, label="Silhouette Score")
    plt.plot(evaluations[:, 0], label="BCubed Precision")
    plt.plot(evaluations[:, 1], label="BCubed Recall")
    plt.plot(evaluations[:, 2], label="BCubed $F_1$ Score")
    plt.axvline(pos, 0, 1, ls="dashed", c="r", label="Max silhouette score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("img/silhouette_score.png")


if __name__ == '__main__':
    clustering_case()
