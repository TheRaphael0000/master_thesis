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

from rank_list_fusion import fusion_z_score
from rank_list_fusion import fusion_regression_training
from rank_list_fusion import fusion_regression

from evaluate import evaluate_linking
from evaluate import evaluate_clustering

from linking import compute_links
from linking import most_frequent_word

from clustering import supervised_clustering_training
from clustering import supervised_clustering_predict
from clustering import unsupervised_clustering
from clustering import clustering_at_every_n_clusters

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
    # s_curve_c()
    # s_curve_r()

    # degradation()
    # token_vs_lemma()
    # letter_ngrams()
    letter_ngrams_2()
    # first_last_letters_ngrams()
    # pos_ngrams()

    # compression_evaluation()

    # frequent_errors()
    # dates_differences()
    # fusion_evaluation()
    # fusion_with_soft_veto()
    # every_fusion()

    # unsupervised_clustering_evaluation()
    # supervised_clustering_evaluation()
    pass


def distance_over_rank():
    _, _, X, _ = brunet.parse()

    rank_list = compute_links([X, 0, 500, False, 0.1, distances.manhattan])
    plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(range(len(rank_list)), [r[-1] for r in rank_list])
    plt.xlabel("Rank")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig("img/distance_over_rank.png")


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
    plt.tight_layout()
    plt.savefig("img/s_curve_c.png")


def s_curve_r():
    scale = 500
    plt.figure(figsize=(4, 3), dpi=200)

    min_ = 0.1
    max_ = 0.9
    rs = np.arange(min_, max_, 0.001)

    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color", plt.cm.hsv(np.linspace(0, 1, len(rs))))

    for ri in rs:
        x, y = s_curves.sigmoid_reciprocal(r=ri)(scale)
        plt.plot(x, y, linewidth=0.2)

    cbar = plt.colorbar(plt.cm.ScalarMappable(
        norm=colors.Normalize(min_, max_), cmap="hsv"))
    cbar.set_label("r")
    plt.tight_layout()
    plt.savefig("img/s_curve_r.png")


def degradation():
    print("loading")
    _, _, _, X, Y = st_jean.parse()

    M = []

    sizes = np.arange(9000, 0, -250, dtype=int)

    for i in sizes:
        # limitate the data size
        Xi = [x[:i] for x in X]
        rl = compute_links([Xi, 0, 500, True, 0.1, distances.cosine_distance])
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


def token_vs_lemma():
    _, _, X_lemma, X_token, Y = st_jean.parse()
    # _, X_lemma, X_token, Y = brunet.parse()

    M_token = []
    M_lemma = []

    mfws = np.arange(100, 2000 + 1, 100)
    distances_ = distances.vector_distances

    for mfw in mfws:
        for zscore, distance in distances_:
            print(mfw)
            rl_token = compute_links([X_token, 0, mfw, zscore, 1e-1, distance])
            Mi = evaluate_linking(rl_token, Y)
            M_token.append(Mi)

            rl_lemma = compute_links([X_lemma, 0, mfw, zscore, 1e-1, distance])
            Mi = evaluate_linking(rl_lemma, Y)
            M_lemma.append(Mi)

    M_token = np.array(M_token).reshape(-1, len(distances_), 3)
    M_lemma = np.array(M_lemma).reshape(-1, len(distances_), 3)

    custom_lines = [
        Line2D([0], [0], color="k", lw=2, ls="dotted"),
        Line2D([0], [0], color="k", lw=2, ls="dashed"),
    ] + [Line2D([0], [0], color=f"C{i}", lw=2) for i in range(len(distances_))]
    labels = ["Token", "Lemma"] + [d.__name__ for z, d in distances_]

    plt.figure(figsize=(6, 8), dpi=200)
    for i in range(len(distances_)):
        plt.plot(mfws, M_token[:, i, 0], ls="dashed", c=f"C{i}")
        plt.plot(mfws, M_lemma[:, i, 0], ls="dotted", c=f"C{i}")
    plt.xlabel("#MFW")
    plt.ylabel("Average Precision (AP)")
    plt.legend(custom_lines, labels, loc="lower center", ncol=2)
    plt.tight_layout()
    plt.grid()
    plt.savefig("img/token_vs_lemma.png")


def letter_ngrams():
    print("loading")
    _, _, _, X, Y = st_jean.parse()
    # _, _, X, Y = brunet.parse()
    # _, X, Y = oxquarry.parse()

    M = defaultdict(list)

    mfws = np.arange(500, 15000 + 1, 500)

    ngrams_types = [3, 4, 5, (2, 3), (3, 4), (4, 5)]
    for ngrams_type in ngrams_types:
        print(ngrams_type)
        for mfw in mfws:
            rep = [X, ngrams_type, mfw, True, 1e-1, distances.cosine_distance]
            rl = compute_links(rep)
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

def letter_ngrams_2():
    print("loading")
    # _, _, _, X, Y = st_jean.parse()
    _, _, X, Y = brunet.parse()
    # _, X, Y = oxquarry.parse()

    configurations = [
        (3, 3000),
        (4, 8000),
    ]

    for n_grams_type, mfw in configurations:
        for zscore, distance in distances.vector_distances:
            rep = [X, n_grams_type, mfw, zscore, 1e-1, distance]
            rl = compute_links(rep)
            m = evaluate_linking(rl, Y)
            print(n_grams_type, mfw, distance.__name__, m)


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
            rep = [word_ngrams_X, 0, mfw, True,
                   1e-1, distances.cosine_distance]
            rl = compute_links()
            m = evaluate_linking(rl, Y)
            M_ngrams.append(m[0])
            rep = [word_begin_X, 0, mfw, True, 1e-1, distances.cosine_distance]
            rl = compute_links(rep)
            m = evaluate_linking(rl, Y)
            M_first.append(m[0])
            rep = [word_end_X, 0, mfw, True, 1e-1, distances.cosine_distance]
            rl = compute_links(rep)
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

    plt.legend(custom_lines, ["n = 3", "n = 4", "n = 5", "word n-grams",
                              "n first letters", "n last letters"], loc="lower right")
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
            rep = [X, ngrams_type, mfw, True, 1e-1, distances.cosine_distance]
            rl = compute_links(rep)
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
            rep = (X, compression_method, distance_func)
            rl = compute_links(rep)
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
    adjust_text(texts, arrowprops=dict(arrowstyle="", color="C0"))
    cbar = plt.colorbar()
    plt.xlabel("RPrec")
    plt.ylabel("Average precision (AP)")
    cbar.set_label("HPrec")
    plt.tight_layout()
    plt.savefig("img/compression_evaluation.png")


def frequent_errors():
    print("loading")
    _, X_pos, _, X_token, Y = st_jean.parse()

    trs = tr(X_token, X_pos)

    top_n = 30
    keep = 2

    incorrectly_ranked = defaultdict(lambda: 0)

    rls = []

    for t in trs:
        rl = compute_links(t)
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

    features, mfw = most_frequent_word(X_token, 750, lidstone_lambda=1e-1)

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

    (a, b), score = rl[int(m[-1] - 1)]
    plot(a, b, f"img/mfw_vector_first_last_rl.png")

    for i, ((a, b), errors) in enumerate(top_errors):
        plot(a, b, f"img/mfw_vector_error_{i}.png")

    (a, b), score = rl[-1]
    plot(a, b, f"img/mfw_vector_last_rl.png")


def dates_differences():
    print("loading")
    info, X_pos, _, X_token, Y = st_jean.parse()

    s = 5

    dates = [int(i[-1]) for i in info]
    plt.figure(figsize=(4, 3), dpi=200)
    plt.hist(dates, bins=np.arange(
        np.min(dates), np.max(dates), s), density=True, alpha=0.7)
    plt.xlabel("Date")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("img/dates_distribution.png")

    trs = tr(X_token, X_pos)

    rls = []
    for t in trs:
        rl = compute_links(t)
        rls.append(rl)
        print(evaluate_linking(rl, Y))
    rl = fusion_z_score(rls)
    print(evaluate_linking(rl, Y))

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


def tr9(X_token, X_pos):
    return [
        [X_token, 0, 750, True, 1e-1, distances.cosine_distance],
        [X_token, 0, 750, False, 1e-1, distances.clark],
        [X_token, 0, 750, True, 1e-1, distances.manhattan],
        [X_token, 0, 750, False, 1e-1, distances.tanimoto],
        [X_token, 3, 3000, True, 1e-1, distances.cosine_distance],
        [X_token, 4, 8000, True, 1e-1, distances.cosine_distance],
        [X_pos, 2, 250, True, 1e-1, distances.cosine_distance],
        [X_pos, 3, 1000, True, 1e-1, distances.cosine_distance],
        (X_token, compressions.bz2, distances.cbc)
    ]


def tr7(X_token):
    return [
        [X_token, 0, 750, True, 1e-1, distances.cosine_distance],
        [X_token, 0, 750, False, 1e-1, distances.clark],
        [X_token, 0, 750, True, 1e-1, distances.manhattan],
        [X_token, 0, 750, False, 1e-1, distances.tanimoto],
        [X_token, 3, 3000, True, 1e-1, distances.cosine_distance],
        [X_token, 4, 8000, True, 1e-1, distances.cosine_distance],
        (X_token, compressions.bz2, distances.cbc)
    ]


def tr(*X):
    if len(X) == 2:
        return tr9(X[0], X[1])
    else:
        return tr7(X[0])


def fusion_evaluation():
    # _, X1, Y1 = oxquarry.parse()
    # _, _, X2, Y2 = brunet.parse()
    _, X3_pos, X3_lemma, X3_token, Y3 = st_jean.parse_A()
    _, X4_pos, X4_lemma, X4_token, Y4 = st_jean.parse_B()

    X_training, Y_training = (X4_token, X4_pos), Y4
    X_testing, Y_testing = (X3_token, X3_pos), Y3

    fusion_size = 4

    models = []
    print("Training rank lists")
    tr_training = tr9(*X_training)[0:5]
    for i, t in enumerate(tr_training):
        rl = compute_links(t)
        model, rmse = fusion_regression_training(rl, Y_training)
        models.append(model)
        mesures = evaluate_linking(rl, Y_training)
        print(i, *mesures, rmse)

    M_single = []
    rank_lists = []
    print("Testing rank lists")
    tr_testing = tr9(*X_testing)[0:5]
    for i, t in enumerate(tr_testing):
        rl = compute_links(t)
        rank_lists.append(rl)
        mesures = evaluate_linking(rl, Y_testing)
        M_single.append(mesures)
        print(i, *mesures)

    M_single = np.array(M_single)

    M_single_max = []
    M_single_mean = []
    M_fusion_z_score = []
    M_fusion_regression = []

    tr_ids = np.array(
        list(itertools.combinations(range(len(tr_training)), fusion_size)))

    for tr_id in tr_ids:
        rls = [rank_lists[i] for i in tr_id]

        m_single_max = np.max(M_single[tr_id, :], axis=0)
        M_single_max.append(m_single_max)

        m_single_mean = np.mean(M_single[tr_id, :], axis=0)
        M_single_mean.append(m_single_mean)

        rl_z_score = fusion_z_score(rls)
        m_z_score = evaluate_linking(rl_z_score, Y_testing)
        M_fusion_z_score.append(m_z_score)

        rl_regression = fusion_regression(models, rls)
        m_regression = evaluate_linking(rl_regression, Y_testing)
        M_fusion_regression.append(m_regression)

    M_single_max = np.array(M_single_max)
    M_single_mean = np.array(M_single_mean)
    M_fusion_z_score = np.array(M_fusion_z_score)
    M_fusion_regression = np.array(M_fusion_regression)

    plt.figure(figsize=(6, 4), dpi=200)
    x, y, c = M_single[:, 1], M_single[:, 0], M_single[:, 2]
    plt.scatter(x, y, c=c, marker="o", label="Single rank list", alpha=0.8)
    x, y, c = M_fusion_regression[:,
                                  1], M_fusion_regression[:, 0], M_fusion_regression[:, 2]
    plt.scatter(x, y, c=c, marker="x",
                label=f"Regression fusions ({fusion_size} lists)", alpha=0.5)
    x, y, c = M_fusion_z_score[:,
                               1], M_fusion_z_score[:, 0], M_fusion_z_score[:, 2]
    plt.scatter(x, y, c=c, marker="+",
                label=f"Z-score fusions ({fusion_size} lists)", alpha=0.5)
    cbar = plt.colorbar()
    plt.xlabel("RPrec")
    plt.ylabel("Average precision (AP)")
    cbar.set_label("HPrec")
    plt.legend()
    plt.tight_layout()
    plt.savefig("img/fusion_evaluation.png")

    print("Fusion Statistics")

    def print_statistics_latex(M_list):
        print("Min &", " & ".join(
            np.round(M_list.min(axis=0), 3).astype(str)), r"\\")
        mean_std = zip(np.round(M_list.mean(axis=0), 3).astype(
            str), np.round(M_list.std(axis=0), 3).astype(str))
        mean_std = [f"{mean}\pm{std}" for mean, std in mean_std]
        print("Mean$\pm$Std &", " & ".join(mean_std), r"\\")
        print("Max &", " & ".join(
            np.round(M_list.max(axis=0), 3).astype(str)), r"\\")
        argmin = tr_ids[np.argmin(M_list, axis=0)]
        print("Argmin &", " & ".join(
            [np.array2string(a, separator=",") for a in argmin]), r"\\")
        argmax = tr_ids[np.argmax(M_list, axis=0)]
        print("Argmax &", " & ".join(
            [np.array2string(a, separator=",") for a in argmax]), r"\\")

    print("Single mean")
    print_statistics_latex(M_single_mean)
    print("Single max")
    print_statistics_latex(M_single_max)
    print("Z-score")
    print_statistics_latex(M_fusion_z_score)
    print("Regression")
    print_statistics_latex(M_fusion_regression)

    print("Fusion sign tests")
    print("Z-score/T/Single-mean")
    print(*sign_test(M_fusion_z_score, M_single_mean))
    print("Z-score/T/Single-max")
    print(*sign_test(M_fusion_z_score, M_single_max))
    print("Regression/T/Single-mean")
    print(*sign_test(M_fusion_regression, M_single_mean))
    print("Regression/T/Single-max")
    print(*sign_test(M_fusion_regression, M_single_max))


def fusion_with_soft_veto():
    print("Loading")
    _, X_oxquarry, Y_oxquarry = oxquarry.parse()
    _, _, X_brunet, Y_brunet = brunet.parse()
    _, X_pos_st_jean_A, _, X_token_st_jean_A, Y_st_jean_A = st_jean.parse_A()
    _, X_pos_st_jean_B, _, X_token_st_jean_B, Y_st_jean_B = st_jean.parse_B()

    datasets = [
        (([X_oxquarry, ], Y_oxquarry), "Oxquarry"),
        (([X_brunet, ], Y_brunet), "Brunet"),
        (([X_token_st_jean_A, X_pos_st_jean_A], Y_st_jean_A), "St-Jean A"),
        (([X_token_st_jean_B, X_pos_st_jean_B], Y_st_jean_B), "St-Jean B"),
    ]

    (X, Y), name = datasets[0]

    rls = []
    tr_ = tr(*X)

    print("linking")
    for i, t in enumerate(tr_):
        rl = compute_links(t)
        mesures = evaluate_linking(rl, Y)
        print(i, *mesures)
        rls.append(rl)

    print("vanilla")
    rl = fusion_z_score(rls)
    M_vanilla = evaluate_linking(rl, Y)[0]

    print("soft veto")
    resolution = 25
    cs = np.linspace(0, 0.1, resolution)
    rs = np.linspace(0, 0.3, resolution)
    print(cs)
    print(rs)
    c_r = np.array(list(itertools.product(cs, rs)))
    M_softveto = []
    for a, b in c_r:
        s_curve = s_curves.full_boost(top=a, bottom=b)
        # s_curve = s_curves.sigmoid_reciprocal(c=a, r=b)
        rls_veto = [s_curves.soft_veto(rl, s_curve) for rl in rls]
        rl = fusion_z_score(rls_veto)
        M_softveto.append(evaluate_linking(rl, Y)[0])

    M_softveto = np.array(M_softveto).reshape((resolution, -1))
    M_softveto -= M_vanilla

    vmax = np.max(np.abs([np.min(M_softveto), np.max(M_softveto)]))

    print(np.max(M_softveto))
    print(c_r[np.argmax(M_softveto)])

    plt.scatter(x=c_r[:, 0], y=c_r[:, 1],
                c=M_softveto, cmap="RdYlGn", marker="s",
                vmin=-vmax, vmax=vmax
                )
    plt.colorbar()
    plt.xlabel("top")
    plt.ylabel("bottom")
    plt.xticks(np.linspace(np.min(cs), np.max(cs), 6))
    plt.yticks(np.linspace(np.min(rs), np.max(rs), 6))
    plt.tight_layout()
    plt.savefig("img/soft_veto_heat.png")


def every_fusion():
    print("Loading")
    _, X_oxquarry, Y_oxquarry = oxquarry.parse()
    _, _, X_brunet, Y_brunet = brunet.parse()
    _, X_pos_st_jean_A, _, X_token_st_jean_A, Y_st_jean_A = st_jean.parse_A()
    _, X_pos_st_jean_B, _, X_token_st_jean_B, Y_st_jean_B = st_jean.parse_B()

    datasets = [
        ([X_oxquarry, ], Y_oxquarry),
        ([X_brunet, ], Y_brunet),
        ([X_token_st_jean_A, X_pos_st_jean_A], Y_st_jean_A),
        ([X_token_st_jean_B, X_pos_st_jean_B], Y_st_jean_B),
    ]

    for Xs, Ys in datasets:
        tr_ = tr(*Xs)
        rls = []
        for i in tr_:
            rl = compute_links(i)
            print(evaluate_linking(rl, Ys))
            rls.append(rl)
        rl = fusion_z_score(rls)
        print(evaluate_linking(rl, Ys), "(overall)")
        print()


def unsupervised_clustering_evaluation():
    # _, X, Y = oxquarry.parse()
    _, _, X, Y = brunet.parse()
    # _, X_pos, _, X_token, Y = st_jean.parse_A()
    # _, X_pos, _, X_token, Y = st_jean.parse_B()

    # tr = tr9(X_token, X_pos)
    tr = tr7(X)

    ips_stop = 0.1

    print("AP RPrec HPrec")
    rank_lists = [compute_links(t) for t in tr]
    for rank_list in rank_lists:
        print(evaluate_linking(rank_list, Y))

    print("Overall")
    rank_list_overall = fusion_z_score(rank_lists)
    print(evaluate_linking(rank_list_overall, Y))

    labels, silhouette_scores = unsupervised_clustering(
        rank_list_overall, ips_stop=ips_stop, return_scores=True)

    print("bcubed.precision", "bcubed.recall", "bcubed.fscore", "r ratio diff")
    print(evaluate_clustering(Y, labels))

    ns, labels_list = clustering_at_every_n_clusters(rank_list_overall)
    evaluations = np.array([evaluate_clustering(Y, labels)
                            for labels in labels_list])

    n_clusters_found = len(np.unique(labels))
    n_clusters_actual = len(np.unique(Y))

    plt.figure(figsize=(6, 4), dpi=200)
    plt.plot(ns, evaluations[:, 0], label="BCubed $F_1$ Score")
    plt.plot(ns, evaluations[:, 1], label="BCubed Precision")
    plt.plot(ns, evaluations[:, 2], label="BCubed Recall")
    plt.plot(ns, evaluations[:, 3], label="r ratio diff")
    plt.plot(*silhouette_scores, label="Silhouette Score")
    plt.axvline(n_clusters_found, 0, 1,
                ls="dashed", c="C4", label="IPS Procedure #Clusters")
    plt.axvline(n_clusters_actual, 0, 1,
                ls="dashed", c="C2", label="Actual #Clusters")
    xmin, xmax, ymin, ymax = plt.axis()
    ypos = ymax / 2 - ymin / 2
    plt.text(n_clusters_found, ypos,
             f"{n_clusters_found}", c="C4", rotation="vertical")
    plt.text(n_clusters_actual, ypos,
             f"{n_clusters_actual}", c="C2", rotation="vertical")
    plt.legend(loc="upper right")
    plt.xlabel("#Clusters")
    plt.ylabel("Metric")
    plt.grid()
    plt.tight_layout()
    plt.savefig("img/unsupervised_clustering.png")


def supervised_clustering_evaluation():
    def plot(rl, Y, n_clusters_found, n_clusters_actual, filename):
        ns, labels_list = clustering_at_every_n_clusters(rl)
        evaluations = np.array([evaluate_clustering(Y, labels)
                                for labels in labels_list])

        plt.figure(figsize=(6, 4), dpi=200)
        plt.plot(ns, evaluations[:, 0], label="BCubed Precision")
        plt.plot(ns, evaluations[:, 1], label="BCubed Recall")
        plt.plot(ns, evaluations[:, 2], label="BCubed $F_1$ Score")
        plt.axvline(n_clusters_found, 0, 1,
                    ls="dashed", c="C3", label="LogisticRegression #Clusters")
        plt.axvline(n_clusters_actual, 0, 1,
                    ls="dashed", c="C2", label="Actual #Clusters")
        xmin, xmax, ymin, ymax = plt.axis()
        ypos = ymax / 2 - ymin / 2
        plt.text(n_clusters_found, ypos,
                 f"{n_clusters_found}", c="C3", rotation="vertical")
        plt.text(n_clusters_actual, ypos,
                 f"{n_clusters_actual}", c="C2", rotation="vertical")
        plt.grid()
        plt.legend(loc="upper right")
        plt.xlabel("#Clusters")
        plt.ylabel("Metric")
        plt.tight_layout()
        plt.savefig(f"img/{filename}.png")

    print("Loading")
    _, X_oxquarry, Y_oxquarry = oxquarry.parse()
    _, _, X_brunet, Y_brunet = brunet.parse()
    _, X_pos_st_jean_A, _, X_token_st_jean_A, Y_st_jean_A = st_jean.parse_A()
    _, X_pos_st_jean_B, _, X_token_st_jean_B, Y_st_jean_B = st_jean.parse_B()

    datasets = [
        ([X_oxquarry, ], Y_oxquarry),
        ([X_brunet, ], Y_brunet),
        ([X_token_st_jean_A, X_pos_st_jean_A], Y_st_jean_A),
        ([X_token_st_jean_B, X_pos_st_jean_B], Y_st_jean_B),
    ]
    dataset_labels = [
        "oxquarry",
        "brunet",
        "st_jean_a",
        "st_jean_b",
    ]
    ids = range(len(datasets))

    dataset_rls = []

    print("Computing rank lists")
    for id in ids:
        X, Y = datasets[id]
        tr_training = tr(*X)
        rls = [compute_links(t) for t in tr_training]
        for rl in rls:
            M = evaluate_linking(rl, Y)
            print(M)
        rl = fusion_z_score(rls)
        dataset_rls.append(rl)
        print(dataset_labels[id], evaluate_linking(rl, Y))

    m = []
    print("Supervised clustering")
    for A, B in itertools.product(ids, ids):
        X_training, Y_training = datasets[A]
        rl_training = dataset_rls[A]
        X_testing, Y_testing = datasets[B]
        rl_testing = dataset_rls[B]

        model = supervised_clustering_training(rl_training, Y_training)
        Y_pred = supervised_clustering_predict(model, rl_testing)

        M = evaluate_clustering(Y_testing, Y_pred)
        m.append(M)
        print(dataset_labels[A], dataset_labels[B], M)

    print(np.array(m).mean(axis=0))


if __name__ == "__main__":
    np.set_printoptions(precision=2, floatmode="fixed")
    main()
