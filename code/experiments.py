import itertools
from collections import defaultdict
from collections import Counter
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon
from adjustText import adjust_text

import distances
import compressions
import s_curves

from corpus import brunet
from corpus import oxquarry
from corpus import st_jean
from corpus import pan16

from rank_list_fusion import fusion_s_curve_score
from rank_list_fusion import fusion_z_score

from evaluate import evaluate_linking

from linking import compute_links
from linking import compute_links_compress
from linking import most_frequent_word

from misc import sign_test
from misc import simple_plot
from misc import first_letters_cut
from misc import word_n_grams
from misc import last_letters_cut
from misc import sigmoid
from misc import sigmoid_r
from misc import compute_r
from misc import normalize


def main():
    # distance_over_rank()
    # s_curve_r()
    # s_curve_c()
    # sigmoids()

    # degradation()
    # mfw()
    # letter_ngrams()
    # first_last_letters_ngrams()
    # pos_ngrams()

    # compression_evaluation()

    # frequent_errors()
    # dates_differences()
    fusion()

    # count_ngrams()
    pass


def compression_evaluation():
    _, _, _, X, Y = st_jean.parse()
    # _, _, X, Y = brunet.parse()
    # _, X, Y = oxquarry.parse()

    compression_methods = [
        compressions.lzma,
        compressions.bz2,
        compressions.gzip,
    ]
    distance_funcs = [
        distances.ncd,
        distances.cbc,
    ]
    distances_compressions = list(itertools.product(
        compression_methods, distance_funcs))

    M = []
    T = []

    for i in range(3):
        for compression_method, distance_func in distances_compressions:
            print(compression_method.__name__, distance_func.__name__)
            t0 = time.time()
            rl = compute_links_compress(X, compression_method, distance_func)
            t = time.time() - t0
            m = evaluate_linking(rl, Y)
            M.append(m)
            T.append(t)
            print(m, t)

    M = np.array(M).reshape(-1, len(distances_compressions), 3)
    T = np.array(T).reshape(-1, len(distances_compressions))
    M = M.mean(axis=0)
    T = T.mean(axis=0)

    print(M)
    print(T)

    plt.figure(figsize=(6, 4), dpi=200)
    x, y, c = M[:, 1], M[:, 0], M[:, 2]
    plt.scatter(x, y, c=c, marker=".")
    texts = []
    for i, (compression_method, distance_func) in enumerate(distances_compressions):
        text = f"({compression_method.__name__}, {distance_func.__name__})"
        xy = (x[i], y[i])
        texts.append(plt.annotate(text, xy))
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="C0"))
    cbar = plt.colorbar()
    plt.xlabel("RPrec")
    plt.ylabel("Average precision (AP)")
    cbar.set_label("HPrec")
    plt.tight_layout()
    plt.savefig("img/compression_evaluation.png")


def dates_differences():
    info, _, _, X, Y = st_jean.parse()

    s = 5

    dates = [int(i[-1]) for i in info]
    plt.figure(figsize=(4, 3), dpi=200)
    plt.hist(dates, bins=np.arange(
        np.min(dates), np.max(dates), s), density=True, alpha=0.7)
    plt.xlabel("Date")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("img/dates_distribution.png")

    experiments = [
        [X, 0, 500, True, 1e-1, distances.manhattan],
        [X, 0, 500, False, 1e-1, distances.tanimoto],
        [X, 0, 500, False, 1e-1, distances.clark],
        [X, 0, 500, False, 1e-1, distances.matusita],
        [X, 0, 500, True, 1e-1, distances.cosine_distance],

        # [X, 6, 500, True, 1e-1, distances.manhattan],
        # [X, 6, 500, False, 1e-1, distances.tanimoto],
        # no clark since too bad results
        # [X, 6, 500, False, 1e-1, distances.matusita],
        # [X, 6, 500, True, 1e-1, distances.cosine_distance],
    ]

    s_curve = s_curves.sigmoid_reciprocal()
    rls = [compute_links(*e) for e in experiments]
    rl = fusion_s_curve_score(rls, s_curve)
    print(*evaluate_linking(rl, Y))

    date_diffs = np.array([np.abs(dates[a] - dates[b]) for (a, b), s in rl])

    correct = np.array([Y[a] == Y[b] for (a, b), s in rl])

    r = compute_r(Y)
    all_links = date_diffs
    true_links = date_diffs[correct]
    top_r_true_links = true_links[0:r]
    false_links = date_diffs[~correct]
    top_r_false_links = false_links[0:r]

    def plot(data, title, color, filename):
        plt.figure(figsize=(4, 3), dpi=200)
        n, bins, patches = plt.hist(data, bins=np.arange(
            0, np.max(data), s), color=color, alpha=0.7, density=True, label="Distibution")
        mean = data.mean()
        std = data.std()
        plt.axvline(mean, c=color, linestyle="dashed",
                    label=f"Mean = {mean:.2f}")
        plt.hlines(y=n.max() / 2 - n.min() / 2, xmin=mean - std // 2, xmax=mean +
                   std // 2, color=color, linestyle="solid", label=f"Std = {std:.2f}")
        h, l = plt.gca().get_legend_handles_labels()
        order = [1, 0, 2]
        plt.legend([h[i] for i in order], [l[i] for i in order])
        plt.xlabel("Date difference")
        plt.ylabel("Density")
        plt.tight_layout()
        xticks = np.arange(date_diffs.min(), date_diffs.max(), 10)
        plt.xticks(xticks)
        plt.savefig(filename)

    plot(all_links, "All links", "C0", "img/dates_differences_all.png")
    plot(false_links, "false links", "C1", "img/dates_differences_false.png")
    plot(top_r_true_links, "top-r true links, true links", "C2",
         "img/dates_differences_r_true.png")
    plot(top_r_false_links, "top-r false links", "C3",
         "img/dates_differences_r_false.png")


def frequent_errors():
    print("loading")
    _, _, _, X, Y = st_jean.parse()

    experiments = [
        [X, 0, 500, True, 1e-1, distances.manhattan],
        [X, 0, 500, False, 1e-1, distances.tanimoto],
        [X, 0, 500, False, 1e-1, distances.clark],
        [X, 0, 500, False, 1e-1, distances.matusita],
        [X, 0, 500, True, 1e-1, distances.cosine_distance],
    ]

    top_n = 10
    keep = 2

    incorrectly_ranked = defaultdict(lambda: 0)

    rls = []

    for exp in experiments:
        rl = compute_links(*exp)
        rls.append(rl)
        m = evaluate_linking(rl, Y)
        print(m)
        i = 0
        for (a, b), s in rl:
            if Y[a] != Y[b]:
                i += 1
                incorrectly_ranked[(a, b)] += 1
                if i > top_n:
                    break

    rl = fusion_z_score(rls)
    m = evaluate_linking(rl, Y)
    print(m, "(overall)")
    top_errors = Counter(dict(incorrectly_ranked)).most_common(keep)
    print(top_errors)

    features, mfw = most_frequent_word(X, 500, lidstone_lambda=1e-1)

    def plot(a, b, filename):
        A, B = features[a, :], features[b, :]
        mean = np.mean(np.array([A, B]), axis=0)
        order_indices = np.argsort(mean)[::-1]
        A = A[order_indices]
        B = B[order_indices]
        plt.figure(figsize=(4, 3), dpi=200)
        plt.yscale("log")
        plt.bar(range(len(A)), A, width=1, label=f"{Y[a]} ({a+1})", alpha=0.5)
        plt.bar(range(len(A)), B, width=1, label=f"{Y[b]} ({b+1})", alpha=0.5)
        plt.legend()
        plt.xticks([], [])
        plt.xlabel("MFW Vector")
        plt.ylabel("Relative word frequency")
        plt.tight_layout()
        plt.savefig(filename)

    (a, b), score = rl[0]
    plot(a, b, f"img/mfw_vector_first_rl.png")

    (a, b), score = rl[m[-1] - 1]
    plot(a, b, f"img/mfw_vector_first_last_rl.png")

    for i, ((a, b), errors) in enumerate(top_errors):
        plot(a, b, f"img/mfw_vector_error_{i}.png")

    (a, b), score = rl[-1]
    plot(a, b, f"img/mfw_vector_last_rl.png")


def first_last_letters_ngrams():
    print("loading")
    # _, _, _, X, Y = st_jean.parse()
    _, _, X, Y = brunet.parse()
    # _, X, Y = oxquarry.parse()


    plt.figure(figsize=(6, 4), dpi=200)

    for n, c in zip([3, 4, 5], ["C0", "C1", "C2"]):
        print(n)
        word_begin_X = first_letters_cut(X, n)
        word_ngrams_X = word_n_grams(X, n)
        word_end_X = last_letters_cut(X, n)

        M_ngrams = []
        M_first = []
        M_last = []

        mfws = np.arange(200, 4000 + 1, 100)

        for mfw in mfws:
            print(mfw)
            rl = compute_links(word_ngrams_X, 0, mfw, True, 1e-1, distances.cosine_distance)
            m = evaluate_linking(rl, Y)
            M_ngrams.append(m[0])
            rl = compute_links(word_begin_X, 0, mfw, True, 1e-1, distances.cosine_distance)
            m = evaluate_linking(rl, Y)
            M_first.append(m[0])
            rl = compute_links(word_end_X, 0, mfw, True, 1e-1, distances.cosine_distance)
            m = evaluate_linking(rl, Y)
            M_last.append(m[0])

        plt.plot(mfws, M_ngrams, c=c, ls="solid")
        plt.plot(mfws, M_first, c=c, ls="dotted")
        plt.plot(mfws, M_last, c=c, ls="dashed")

    custom_lines = [
        Line2D([0], [0], color="C0", lw=2),
        Line2D([0], [0], color="C1", lw=2),
        Line2D([0], [0], color="C2", lw=2),
        Line2D([0], [0], color="k", lw=2, ls="solid"),
        Line2D([0], [0], color="k", lw=2, ls="dotted"),
        Line2D([0], [0], color="k", lw=2, ls="dashed"),
    ]

    plt.legend(custom_lines, ["n = 3", "n = 4", "n = 5", "word n-grams", "n first letters", "n last letters"], loc="lower right")
    plt.xlabel("MFW")
    plt.ylabel("Average Precision (AP)")
    plt.tight_layout()
    plt.savefig("img/first_last_letters_ngrams.png")


def pos_ngrams():
    print("loading")
    _, X, _, _, Y = st_jean.parse()

    M = defaultdict(list)

    mfws = np.arange(100, 2000 + 1, 100)

    ngrams_types = [2, 3, 4, (2, 3)]
    for ngrams_type in ngrams_types:
        for mfw in mfws:
            rl = compute_links(X, ngrams_type, mfw, True,
                               1e-1, distances.cosine_distance)
            m = evaluate_linking(rl, Y)
            M[ngrams_type].append(m)
            print(mfw, m)

    M = dict(M)

    plt.figure(figsize=(6, 4), dpi=200)
    for ngrams_type in ngrams_types:
        X = mfws
        Y = [i[0] for i in M[ngrams_type]]
        plt.plot(X, Y, label=f"POS {str(ngrams_type)}-grams")
    plt.legend()
    plt.xlabel("MFW")
    plt.ylabel("Average Precision (AP)")
    plt.tight_layout()
    plt.savefig("img/pos_ngrams.png")


def letter_ngrams():
    print("loading")
    _, _, _, X, Y = st_jean.parse()

    M = defaultdict(list)

    mfws = np.arange(100, 2000 + 1, 100)

    ngrams_types = [2, 3, 4, (2, 3)]
    for ngrams_type in ngrams_types:
        for mfw in mfws:
            rl = compute_links(X, ngrams_type, mfw, True,
                               1e-1, distances.cosine_distance)
            m = evaluate_linking(rl, Y)
            M[ngrams_type].append(m)
            print(mfw, m)

    M = dict(M)

    plt.figure(figsize=(6, 4), dpi=200)
    for ngrams_type in ngrams_types:
        X = mfws
        Y = [i[0] for i in M[ngrams_type]]
        plt.plot(X, Y, label=f"Letters {str(ngrams_type)}-grams")
    plt.legend()
    plt.xlabel("MFW")
    plt.ylabel("Average Precision (AP)")
    plt.tight_layout()
    plt.savefig("img/letter_ngrams.png")


def degradation():
    print("loading")
    _, _, _, X, Y = st_jean.parse()

    M = []

    sizes = np.arange(9000, 0, -250, dtype=int)

    for i in sizes:
        # limitate the data size
        Xi = [x[:i] for x in X]
        rl = compute_links(Xi, 0, 500, True, 0.1, distances.cosine_distance)
        m = evaluate_linking(rl, Y)
        print(i, m)
        M.append(m)

    M = np.array(M)

    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=200)
    ax2 = ax1.twinx()
    ax1.plot(sizes, M[:, 0], c="C0", ls="solid",
             label="Average Precision (AP)")
    ax1.plot(sizes, M[:, 1], c="C0", ls="dashed", label="RPrec")
    ax2.plot(sizes, M[:, 2], c="C0", ls="dotted", label="HPRec")
    ax1.set_xlabel("#Tokens per texts")
    plt.gca().invert_xaxis()
    ax1.set_ylabel("AP/RPrec")
    ax2.set_ylabel("HPrec")
    plt.xticks(np.arange(9000, -1, -1000, dtype=int))
    fig.legend()
    plt.tight_layout()
    plt.savefig("img/degradation.png")


def s_curve_c():
    scale = 500
    plt.figure(figsize=(4, 3), dpi=200)

    min_ = 1e-10
    max_ = 20
    zoom_factors = np.arange(min_, max_, 0.06)

    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color", plt.cm.hsv(np.linspace(0, 1, len(zoom_factors))))

    for i in zoom_factors:
        x, y = s_curves.sigmoid_reciprocal(c=i, r=0.5)(scale)
        plt.plot(x, y, linewidth=0.2)

    cbar = plt.colorbar(plt.cm.ScalarMappable(
        norm=colors.Normalize(min_, max_), cmap="hsv"))
    cbar.set_label("c")
    # plt.legend()
    plt.tight_layout()
    plt.savefig("img/s_curve_c.png")


def s_curve_r():
    scale = 500
    plt.figure(figsize=(4, 3), dpi=200)

    for ri in [0.25, 0.5, 0.75]:
        x, y = s_curves.sigmoid_reciprocal(r=ri)(scale)
        plt.plot(x, y, label=f"r = {ri}")

    plt.legend()
    plt.tight_layout()
    plt.savefig("img/s_curve_r.png")


def distance_over_rank():
    _, _, X, _ = brunet.parse()

    rank_list = compute_links(X, 0, 500, False, 0.1, distances.manhattan)
    plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(range(len(rank_list)), [r[-1] for r in rank_list])
    plt.xlabel("Rank")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig("img/distance_over_rank.png")


def mfw():
    _, _, X, Y = brunet.parse()
    # _, _, _, X, Y = st_jean.parse()
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
    plt.savefig("img/mfw.png")


def fusion():
    _, _, _, X, Y = st_jean.parse()
    # _, _, X, Y = brunet.parse()
    # _, X, Y = oxquarry.parse()

    experiments = [
        [X, 0, 500, True, 1e-1, distances.manhattan],
        [X, 0, 500, False, 1e-1, distances.tanimoto],
        [X, 0, 500, False, 1e-1, distances.clark],
        [X, 0, 500, False, 1e-1, distances.matusita],
        [X, 0, 500, True, 1e-1, distances.cosine_distance],
        [X, 0, 500, False, 1e-1, distances.kld],

        [X, 3, 750, True, 1e-1, distances.manhattan],
        [X, 3, 750, False, 1e-1, distances.tanimoto],
        [X, 3, 750, False, 1e-1, distances.clark],
        [X, 3, 750, False, 1e-1, distances.matusita],
        [X, 3, 750, True, 1e-1, distances.cosine_distance],
        [X, 3, 750, False, 1e-1, distances.kld],

        # (X, compressions.lzma, distances.ncd),
        # (X, compressions.lzma, distances.cbc),
        # (X, compressions.bz2, distances.ncd),
        # (X, compressions.bz2, distances.cbc),
    ]
    s_curve = s_curves.sigmoid_reciprocal()

    fusion_size = 4

    M_single = []

    rank_lists = []
    for i, e in enumerate(experiments):
        if type(e) == tuple:
            rl = compute_links_compress(*e)
        else:
            rl = compute_links(*e)
        rank_lists.append(rl)
        mesures = evaluate_linking(rl, Y)
        print(i, mesures)
        M_single.append(mesures)

    M_single = np.array(M_single)

    M_single_max_in_exp = []
    M_fusion_s_curve = []
    M_fusion_z_score = []

    experiments_ids = np.array(list(itertools.combinations(range(len(experiments)), fusion_size)))

    for experiments_id in experiments_ids:
        rls = [rank_lists[i] for i in experiments_id]
        M_single_max_in_exp.append(np.max(M_single[experiments_id, :], axis=0))

        rl_s_curve = fusion_s_curve_score(rls, s_curve)
        M_fusion_s_curve.append(evaluate_linking(rl_s_curve, Y))

        rl_z_score = fusion_z_score(rls)
        M_fusion_z_score.append(evaluate_linking(rl_z_score, Y))

    M_fusion_s_curve = np.array(M_fusion_s_curve)
    M_fusion_z_score = np.array(M_fusion_z_score)
    M_single_max_in_exp = np.array(M_single_max_in_exp)

    print("S-curve")
    print(M_fusion_s_curve.min(axis=0))
    print(M_fusion_s_curve.max(axis=0))
    print(M_fusion_s_curve.mean(axis=0))
    print(M_fusion_s_curve.std(axis=0))
    print(experiments_ids[np.argmax(M_fusion_s_curve, axis=0)])
    print("Z-score")
    print(M_fusion_z_score.min(axis=0))
    print(M_fusion_z_score.max(axis=0))
    print(M_fusion_z_score.mean(axis=0))
    print(M_fusion_z_score.std(axis=0))
    print(experiments_ids[np.argmax(M_fusion_z_score, axis=0)])
    print("Max in exp")
    print(M_single_max_in_exp.min(axis=0))
    print(M_single_max_in_exp.max(axis=0))
    print(M_single_max_in_exp.mean(axis=0))
    print(M_single_max_in_exp.std(axis=0))
    print(experiments_ids[np.argmax(M_single_max_in_exp, axis=0)])

    print("S-curve vs Single max")
    print(*sign_test(M_fusion_s_curve, M_single_max_in_exp))
    print("Z-score vs Single max")
    print(*sign_test(M_fusion_z_score, M_single_max_in_exp))
    print("S-curve vs Z-score")
    print(*sign_test(M_fusion_s_curve, M_fusion_z_score))

    plt.figure(figsize=(6, 4), dpi=200)
    x, y, c = M_fusion_s_curve[:, 1], M_fusion_s_curve[:, 0], M_fusion_s_curve[:, 2]
    plt.scatter(x, y, c=c, marker="x", label=f"S-curve fusions ({fusion_size} lists)", alpha=0.6)
    x, y, c = M_fusion_z_score[:, 1], M_fusion_z_score[:, 0], M_fusion_z_score[:, 2]
    plt.scatter(x, y, c=c, marker="+", label=f"Z-score fusions ({fusion_size} lists)", alpha=0.6)
    x, y, c = M_single[:, 1], M_single[:, 0], M_single[:, 2]
    plt.scatter(x, y, c=c, marker="o", label="Single rank list", alpha=1)
    cbar = plt.colorbar()
    plt.xlabel("RPrec")
    plt.ylabel("Average precision (AP)")
    cbar.set_label("HPrec")
    plt.legend()
    plt.tight_layout()
    plt.savefig("img/fusion.png")


def sigmoids():
    x = np.linspace(-4, 4, 100)
    y = sigmoid(x)
    plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("$S(x)$")
    plt.tight_layout()
    plt.savefig("img/sigmoid.png")

    x = np.linspace(sigmoid(-4), sigmoid(4), 100)
    y = sigmoid_reciprocal(x)
    plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("$S^{-1}(x)$")
    plt.tight_layout()
    plt.savefig("img/sigmoid_r.png")

def count_ngrams():
    words = open("corpus/english_dictionary.txt").read().split("\n")
    words = [normalize(w) for w in words]

    print(len(Counter(word_n_grams([words], 1)[0])))
    print(len(Counter(word_n_grams([words], 2)[0])))
    print(len(Counter(word_n_grams([words], 3)[0])))
    print(len(Counter(word_n_grams([words], 4)[0])))
    print(len(Counter(word_n_grams([words], 5)[0])))

if __name__ == "__main__":
    main()
