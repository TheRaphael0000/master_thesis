import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from corpus import brunet, oxquarry, st_jean, pan16
from rank_list_fusion import compute_multiple_links, rank_list_fusion
from evaluate import evaluate_linking
from linking import compute_links
import distances
import s_curves


def main():
    # distance_over_rank()
    # zoom_influance()
    # find_best_mfw()
    test_rank_list_fusion()


def zoom_influance():
    scale = 500
    plt.figure(figsize=(4, 3), dpi=200)

    min_ = 1e-10
    max_ = 20
    zoom_factors = np.arange(min_, max_, 0.06)

    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color", plt.cm.hsv(np.linspace(0, 1, len(zoom_factors))))

    for i in zoom_factors:
        x, y = s_curves.sigmoid_reciprocal(i)(scale)
        plt.plot(x, y, linewidth=0.2)

    plt.colorbar(plt.cm.ScalarMappable(
        norm=colors.Normalize(min_, max_), cmap="hsv"))
    # plt.legend()
    plt.tight_layout()
    plt.savefig("zoom_influance.png")


def distance_over_rank():
    _, _, X, _ = brunet.parse()

    rank_list = compute_links(X, 0, 500, False, 0.1, distances.manhattan)
    plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(range(len(rank_list)), [r[-1] for r in rank_list])
    plt.xlabel("Rank")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig("distance_over_rank.png")


def find_best_mfw():
    _, _, X, Y = brunet.parse()
    # _, _, X, Y = st_jean.parse()
    # _, X, Y = oxquarry.parse()

    M = []

    print("AP RPrec P@10 HPrec")
    for mfw in np.arange(100, 2500, 200):
        rank_list = compute_links(X, 0, mfw, False, 0.1, distances.clark)
        mesures = evaluate_linking(rank_list, Y)
        M.append((mfw, *mesures))
        print(mfw, *mesures)

    M = np.array(M)
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), dpi=200)
    axs[0].plot(M[:, 0], M[:, 1])
    axs[1].plot(M[:, 0], M[:, 2])
    axs[2].plot(M[:, 0], M[:, 4])
    plt.tight_layout()
    plt.savefig("mfw.png")


def test_rank_list_fusion():
    # _, _, X, Y = st_jean.parse()
    _, _, X, Y = brunet.parse()
    # _, X, Y = oxquarry.parse()

    experiments = [
        [X, 0, 500, True, 1e-1, distances.manhattan],
        [X, 0, 500, False, 1e-1, distances.tanimoto],
        [X, 0, 500, False, 1e-1, distances.clark],
        [X, 0, 500, False, 1e-1, distances.matusita],
        [X, 0, 500, True, 1e-1, distances.cosine_distance],

        [X, 6, 500, True, 1e-1, distances.manhattan],
        [X, 6, 500, False, 1e-1, distances.tanimoto],
        # no clark since too bad results
        [X, 6, 500, False, 1e-1, distances.matusita],
        [X, 6, 500, True, 1e-1, distances.cosine_distance],
    ]
    s_curve = s_curves.sigmoid_reciprocal()

    M_fusion = []
    M_single = []

    _, rank_lists = compute_multiple_links(experiments, None)
    for rank_list in rank_lists:
        mesures = evaluate_linking(rank_list, Y)
        print(mesures)
        M_single.append(mesures)
    M_single = np.array(M_single)

    pos = np.zeros((len(M_single[0]),))
    eq = np.zeros((len(M_single[0]),))
    neg = np.zeros((len(M_single[0]),))

    for experiments_ in itertools.combinations(range(len(experiments)), 3):
        rank_lists_ = [rank_lists[i] for i in experiments_]
        overall_ranklist = rank_list_fusion(rank_lists_, s_curve)
        mesures = evaluate_linking(overall_ranklist, Y)
        M_fusion.append(mesures)

        for i in range(len(M_single[0])):
            max_ap = max(M_single[experiments_, i])
            if mesures[i] == max_ap:
                eq[i] += 1
            if mesures[i] > max_ap:
                pos[i] += 1
            if mesures[i] < max_ap:
                neg[i] += 1

    print(pos)
    print(eq)
    print(neg)

    M_fusion = np.array(M_fusion)

    plt.figure(figsize=(6, 4), dpi=200)
    x, y, c = M_fusion[:, 0], M_fusion[:, 1], M_fusion[:, 3]
    plt.scatter(x, y, c=c, marker=".", label="Fusions", alpha=0.75)
    x, y, c = M_single[:, 0], M_single[:, 1], M_single[:, 3]
    plt.scatter(x, y, c=c, marker="^", label="Single rank list", alpha=1)
    cbar = plt.colorbar()
    plt.xlabel("HPrec")
    plt.ylabel("Average precision (AP)")
    cbar.set_label("RPrec")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fusion.png")


if __name__ == '__main__':
    main()
