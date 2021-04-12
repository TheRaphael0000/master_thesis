import itertools
from collections import defaultdict
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import wilcoxon
from adjustText import adjust_text

import distances
import compressions
import s_curves

from corpus import brunet
from corpus import oxquarry
from corpus import st_jean
from corpus import pan16

from rank_list_fusion import compute_multiple_links
from rank_list_fusion import rank_list_fusion

from evaluate import evaluate_linking

from linking import compute_links
from linking import compute_links_compress
from linking import most_frequent_word

from misc import sign_test
from misc import simple_plot
from misc import first_letters_cut
from misc import last_letters_cut
from misc import sigmoid
from misc import sigmoid_r
from misc import compute_r


def main():
    # distance_over_rank()
    # s_curve_r()
    # s_curve_c()
    # sigmoids()
    # mfw()
    # fusion()
    # degradation()
    # pos_ngrams()
    # first_last_letters_ngrams()
    # letter_ngrams()
    # recurrent_errors()
    # dates_differences_errors()
    compression_evaluation()
    pass


def compression_evaluation():
    # _, _, _, X, Y = st_jean.parse()
    # _, _, X, Y = brunet.parse()
    _, X, Y = oxquarry.parse()

    compression_methods = [
        compressions.gzip,
        compressions.zlib,
        compressions.bz2,
        compressions.lzma
    ]
    distance_funcs = [
        distances.ncd,
        distances.cbc
    ]
    distances_compressions = list(itertools.product(
        compression_methods, distance_funcs))

    M = []
    for compression_method, distance_func in distances_compressions:
        rl = compute_links_compress(X, compression_method, distance_func)
        m = evaluate_linking(rl, Y)
        M.append(m)
        print(m)

    M = np.array(M)

    plt.figure(figsize=(6, 4), dpi=200)
    plt.title("Compression evaluation")
    x, y, c = M[:, 0], M[:, 1], M[:, 2]
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


def dates_differences_errors():
    info, _, _, X, Y = st_jean.parse()

    s = 5

    dates = [int(i[-1]) for i in info]
    plt.figure(figsize=(4, 3), dpi=200)
    plt.title("Dates distribution")
    plt.hist(dates, bins=np.arange(
        np.min(dates), np.max(dates), s), density=True, alpha=0.7)
    plt.xlabel("Date")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("img/dates_distibution.png")

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
    rl, rls = compute_multiple_links(experiments, s_curve)
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
        plt.title(title)
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


def recurrent_errors():
    print("loading")
    _, _, _, X, Y = st_jean.parse()

    experiments = [
        [X, 0, 500, True, 1e-1, distances.manhattan],
        [X, 0, 500, False, 1e-1, distances.tanimoto],
        [X, 0, 500, False, 1e-1, distances.clark],
        [X, 0, 500, False, 1e-1, distances.matusita],
        [X, 0, 500, True, 1e-1, distances.cosine_distance],

        # [X, 6, 500, True, 1e-1, distances.manhattan],
        # [X, 6, 500, False, 1e-1, distances.tanimoto],
        # # no clark since too bad results
        # [X, 6, 500, False, 1e-1, distances.matusita],
        # [X, 6, 500, True, 1e-1, distances.cosine_distance],
    ]

    top_n = 10
    keep = 3

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

    rl = rank_list_fusion(rls, s_curves.sigmoid_reciprocal())
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
        plt.bar(range(len(A)), A, width=1, label=f"{Y[a]} ({a+1})")
        plt.bar(range(len(A)), B, width=1, label=f"{Y[b]} ({b+1})")
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
    _, _, _, X, Y = st_jean.parse()

    n = 4

    begin_X = first_letters_cut(X, n)
    end_X = last_letters_cut(X, n)

    M_begin = []
    M_end = []
    M_ngrams = []

    mfws = np.arange(100, 2000 + 1, 100)

    ngrams_types = [2, 3, 4, (2, 3)]
    for mfw in mfws:
        print(mfw)
        rl = compute_links(begin_X, 0, mfw, True, 1e-1,
                           distances.cosine_distance)
        m = evaluate_linking(rl, Y)
        M_begin.append(m[0])
        rl = compute_links(end_X, 0, mfw, True, 1e-1,
                           distances.cosine_distance)
        m = evaluate_linking(rl, Y)
        M_end.append(m[0])
        rl = compute_links(X, 4, mfw, True, 1e-1, distances.cosine_distance)
        m = evaluate_linking(rl, Y)
        M_ngrams.append(m[0])

    plt.figure(figsize=(6, 4), dpi=200)
    plt.plot(mfws, M_begin, label="first 4 letters")
    plt.plot(mfws, M_end, label="last 4 letters")
    plt.plot(mfws, M_ngrams, label="4 letters n-grams")
    plt.legend()
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

        [X, 6, 500, True, 1e-1, distances.manhattan],
        [X, 6, 500, False, 1e-1, distances.tanimoto],
        # no clark since too bad results
        [X, 6, 500, False, 1e-1, distances.matusita],
        [X, 6, 500, True, 1e-1, distances.cosine_distance],
    ]
    s_curve = s_curves.sigmoid_reciprocal()
    s_curve_lin = s_curves.linear()

    fusion_size = 3

    M_single = []

    _, rank_lists = compute_multiple_links(experiments, None)
    for rank_list in rank_lists:
        mesures = evaluate_linking(rank_list, Y)
        print(mesures)
        M_single.append(mesures)
    M_single = np.array(M_single)

    M_single_max_in_exp = []
    M_fusion = []
    M_fusion_lin = []

    for experiments_ in itertools.combinations(range(len(experiments)), fusion_size):
        rank_lists_ = [rank_lists[i] for i in experiments_]
        M_single_max_in_exp.append(np.max(M_single[experiments_, :], axis=0))

        overall_ranklist = rank_list_fusion(rank_lists_, s_curve)
        M_fusion.append(evaluate_linking(overall_ranklist, Y))

        overall_ranklist_lin = rank_list_fusion(rank_lists_, s_curve_lin)
        M_fusion_lin.append(evaluate_linking(overall_ranklist_lin, Y))

    M_fusion = np.array(M_fusion)
    M_fusion_lin = np.array(M_fusion_lin)
    M_single_max_in_exp = np.array(M_single_max_in_exp)

    print("S-curve vs Linear")
    print(*sign_test(M_fusion, M_fusion_lin))
    print("S-curve vs Single max")
    print(*sign_test(M_fusion, M_single_max_in_exp))
    print("Linear vs Single max")
    print(*sign_test(M_fusion_lin, M_single_max_in_exp))

    # for i in range(4):
    #     print(M_fusion.shape)
    #     print(wilcoxon(M_fusion[:,i], M_fusion_lin[:,i], alternative="two-sided"))
    #     print(wilcoxon(M_fusion[:,i], M_single_max_in_exp[:,i], alternative="two-sided"))
    #     print(wilcoxon(M_fusion_lin[:,i], M_single_max_in_exp[:,i], alternative="two-sided"))

    plt.figure(figsize=(6, 4), dpi=200)
    x, y, c = M_fusion[:, 0], M_fusion[:, 1], M_fusion[:, 2]
    plt.scatter(x, y, c=c, marker=".",
                label=f"Fusions ({fusion_size} lists)", alpha=0.75)
    x, y, c = M_single[:, 0], M_single[:, 1], M_single[:, 2]
    plt.scatter(x, y, c=c, marker="^", label="Single rank list", alpha=1)
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


if __name__ == "__main__":
    main()
