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
from linking import compute_links_compress
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
    # sigmoids()

    # degradation()
    # token_vs_lemma()
    # letter_ngrams()
    # first_last_letters_ngrams()
    # pos_ngrams()

    # compression_evaluation()

    # frequent_errors()
    # dates_differences()
    fusion()

    # count_ngrams()

    # unsupervised_clustering_evaluation()
    # supervised_clustering_evaluation()


def token_vs_lemma():
    _, _, X_lemma, X_token, Y = st_jean.parse()
    # _, X_lemma, X_token, Y = brunet.parse()

    M_token = []
    M_lemma = []

    mfws = np.arange(100, 2000 + 1, 100)
    distances_ = [
        (True, distances.manhattan),
        (False, distances.tanimoto),
        (True, distances.euclidean),
        (False, distances.matusita),
        (False, distances.clark),
        (True, distances.cosine_distance),
        (False, distances.kld),
        (False, distances.j_divergence),
    ]

    for mfw in mfws:
        for zscore, distance in distances_:
            print(mfw)
            rl_token = compute_links(X_token, 0, mfw, zscore, 1e-1, distance)
            Mi = evaluate_linking(rl_token, Y)
            M_token.append(Mi)

            rl_lemma = compute_links(X_lemma, 0, mfw, zscore, 1e-1, distance)
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

    text_representations = [
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
    rls = [compute_links(*t) for t in text_representations]
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

    text_representations = [
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

    for t in text_representations:
        rl = compute_links(*t)
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
            rl = compute_links(word_ngrams_X, 0, mfw, True,
                               1e-1, distances.cosine_distance)
            m = evaluate_linking(rl, Y)
            M_ngrams.append(m[0])
            rl = compute_links(word_begin_X, 0, mfw, True,
                               1e-1, distances.cosine_distance)
            m = evaluate_linking(rl, Y)
            M_first.append(m[0])
            rl = compute_links(word_end_X, 0, mfw, True,
                               1e-1, distances.cosine_distance)
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
    # _, _, X, Y = brunet.parse()
    # _, X, Y = oxquarry.parse()

    M = defaultdict(list)

    mfws = np.arange(500, 15000 + 1, 500)

    ngrams_types = [3, 4, 5, (2, 3), (3, 4), (4, 5)]
    for ngrams_type in ngrams_types:
        print(ngrams_type)
        for mfw in mfws:
            rep = [X, ngrams_type, mfw, True, 1e-1, distances.cosine_distance]
            rl = compute_links(*rep)
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


def distance_over_rank():
    _, _, X, _ = brunet.parse()

    rank_list = compute_links(X, 0, 500, False, 0.1, distances.manhattan)
    plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(range(len(rank_list)), [r[-1] for r in rank_list])
    plt.xlabel("Rank")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig("img/distance_over_rank.png")


def fusion():
    _, X1, Y1 = oxquarry.parse()
    _, _, X2, Y2 = brunet.parse()
    _, _, _, X3, Y3 = st_jean.parse_A()
    _, _, _, X4, Y4 = st_jean.parse_B()

    X_training, Y_training = X1, Y1
    X_testing, Y_testing = X2, Y2

    def tr(X):
        return [
            [X, 0, 500, True, 1e-1, distances.manhattan],
            [X, 0, 500, False, 1e-1, distances.tanimoto],
            [X, 0, 500, False, 1e-1, distances.clark],
            [X, 0, 500, False, 1e-1, distances.matusita],
            [X, 0, 500, True, 1e-1, distances.cosine_distance],
            [X, 0, 500, False, 1e-1, distances.kld],
            #
            [X, 3, 3000, True, 1e-1, distances.manhattan],
            [X, 3, 3000, False, 1e-1, distances.tanimoto],
            [X, 3, 3000, False, 1e-1, distances.clark],
            [X, 3, 3000, False, 1e-1, distances.matusita],
            [X, 3, 3000, True, 1e-1, distances.cosine_distance],
            [X, 3, 3000, False, 1e-1, distances.kld],
        ]

    fusion_size = 4

    models = []
    for i, t in enumerate(tr(X_training)):
        rl = compute_links(*t)
        model, rmse = fusion_regression_training(rl, Y_training)
        models.append(model)
        mesures = evaluate_linking(rl, Y_training)
        print(i, *mesures, rmse)

    M_single = []
    rank_lists = []
    for i, t in enumerate(tr(X_testing)):
        rl = compute_links(*t)
        rank_lists.append(rl)
        mesures = evaluate_linking(rl, Y_testing)
        M_single.append(mesures)
        print(i, *mesures)

    M_single = np.array(M_single)

    M_single_max = []
    M_fusion_z_score = []
    M_fusion_regression = []

    tr_ids = np.array(
        list(itertools.combinations(range(len(tr(X_training))), fusion_size)))

    for tr_id in tr_ids:
        rls = [rank_lists[i] for i in tr_id]

        m_single_max = np.max(M_single[tr_id, :], axis=0)
        M_single_max.append(m_single_max)

        rl_z_score = fusion_z_score(rls)
        m_z_score = evaluate_linking(rl_z_score, Y_testing)
        M_fusion_z_score.append(m_z_score)

        rl_regression = fusion_regression(models, rls)
        m_regression = evaluate_linking(rl_regression, Y_testing)
        M_fusion_regression.append(m_regression)

    M_single_max = np.array(M_single_max)
    M_fusion_z_score = np.array(M_fusion_z_score)
    M_fusion_regression = np.array(M_fusion_regression)

    print("Single max")
    print(M_single_max.min(axis=0))
    print(M_single_max.max(axis=0))
    print(M_single_max.mean(axis=0))
    print(M_single_max.std(axis=0))
    print(tr_ids[np.argmin(M_single_max, axis=0)])
    print(tr_ids[np.argmax(M_single_max, axis=0)])
    print("Z-score")
    print(M_fusion_z_score.min(axis=0))
    print(M_fusion_z_score.max(axis=0))
    print(M_fusion_z_score.mean(axis=0))
    print(M_fusion_z_score.std(axis=0))
    print(tr_ids[np.argmin(M_fusion_z_score, axis=0)])
    print(tr_ids[np.argmax(M_fusion_z_score, axis=0)])
    print("Regression")
    print(M_fusion_regression.min(axis=0))
    print(M_fusion_regression.max(axis=0))
    print(M_fusion_regression.mean(axis=0))
    print(M_fusion_regression.std(axis=0))
    print(tr_ids[np.argmin(M_fusion_regression, axis=0)])
    print(tr_ids[np.argmax(M_fusion_regression, axis=0)])

    print("Z-score vs Single max")
    print(*sign_test(M_fusion_z_score, M_single_max))
    print("Regression vs Single max")
    print(*sign_test(M_fusion_regression, M_single_max))
    print("Regression vs Z-score")
    print(*sign_test(M_fusion_regression, M_fusion_z_score))

    plt.figure(figsize=(6, 4), dpi=200)
    x, y, c = M_fusion_regression[:,
                                  1], M_fusion_regression[:, 0], M_fusion_regression[:, 2]
    plt.scatter(x, y, c=c, marker="x",
                label=f"Regression fusions ({fusion_size} lists)", alpha=0.6)
    x, y, c = M_fusion_z_score[:,
                               1], M_fusion_z_score[:, 0], M_fusion_z_score[:, 2]
    plt.scatter(x, y, c=c, marker="+",
                label=f"Z-score fusions ({fusion_size} lists)", alpha=0.6)
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

    print(len(words))

    print(len(Counter(word_n_grams([words], 1)[0])))
    print(len(Counter(word_n_grams([words], 2)[0])))
    print(len(Counter(word_n_grams([words], 3)[0])))
    print(len(Counter(word_n_grams([words], 4)[0])))
    print(len(Counter(word_n_grams([words], 5)[0])))


def supervised_clustering_evaluation():
    def linking(X):
        text_representations = [
            [X, 0, 500, True, 1e-1, distances.manhattan],
            [X, 0, 500, False, 1e-1, distances.tanimoto],
            [X, 0, 500, False, 1e-1, distances.clark],
            [X, 0, 500, False, 1e-1, distances.matusita],
            [X, 0, 500, True, 1e-1, distances.cosine_distance],

            [X, 6, 500, True, 1e-1, distances.manhattan],
            [X, 6, 500, True, 1e-1, distances.cosine_distance],
        ]
        rls = [compute_links(*t) for t in text_representations]
        rl = fusion_z_score(rls)
        return rl, rls

    def linking_evaluation(rl, rls, Y):
        print("AP RPrec HPrec (Used for overall)")
        for rl_ in rls:
            print(*evaluate_linking(rl_, Y))
        print("AP RPrec HPrec (Overall)")
        print(*evaluate_linking(rl, Y))

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

    def do(X_training, Y_training, X_testing, Y_testing):
        print("Training")
        rl, rls = linking(X_training)
        linking_evaluation(rl, rls, Y_training)

        print("Learning cut")
        model, eval = supervised_clustering_training(
            rl, Y_training, return_eval=True)
        print(eval)

        print("Testing")
        rl, rls = linking(X_testing)
        linking_evaluation(rl, rls, Y_testing)
        labels = supervised_clustering_predict(model, rl)
        n_clusters_found = len(np.unique(labels))
        n_clusters_actual = len(np.unique(Y_testing))
        plot(rl, Y_testing, n_clusters_found,
             n_clusters_actual, "supervised_clustering_testing")
        M = evaluate_clustering(Y_testing, labels)
        diff = n_clusters_found - n_clusters_actual
        return M, diff

    print("Loading")
    datasets = [
        oxquarry.parse()[-2:],
        brunet.parse()[-2:],
        st_jean.parse_A()[-2:],
        st_jean.parse_B()[-2:]
    ]
    dataset_labels = [
        "oxquarry",
        "brunet",
        "st_jean_a",
        "st_jean_b",
    ]
    ids = range(len(datasets))

    diffs = {}

    # for X, Y in datasets:
    #     rl, rls = linking(X)
    #     linking_evaluation(rl, rls, Y)

    for A, B in itertools.product(ids, ids):
        X_training, Y_training = datasets[A]
        X_testing, Y_testing = datasets[B]
        print(dataset_labels[A], dataset_labels[B])
        diffs[(A, B)] = do(X_training, Y_training, X_testing, Y_testing)

    for A, B in itertools.product(ids, ids):
        print(dataset_labels[A], dataset_labels[B], diffs[(A, B)])


def unsupervised_clustering_evaluation():
    # _, X, Y = oxquarry.parse()
    # _, _, X, Y = brunet.parse()
    _, _, _, X, Y = st_jean.parse_B()

    text_representations = [
        [X, 0, 500, True, 1e-1, distances.manhattan],
        [X, 0, 500, False, 1e-1, distances.tanimoto],
        [X, 0, 500, False, 1e-1, distances.clark],
        [X, 0, 500, False, 1e-1, distances.matusita],
        [X, 0, 500, True, 1e-1, distances.cosine_distance],
        [X, 6, 500, True, 1e-1, distances.manhattan],
        [X, 6, 500, True, 1e-1, distances.cosine_distance],
    ]

    print("AP RPrec HPrec")
    rank_lists = [compute_links(*t) for t in text_representations]
    for rank_list in rank_lists:
        print(evaluate_linking(rank_list, Y))

    print("Overall")
    rank_list_overall = fusion_z_score(rank_lists)
    print(evaluate_linking(rank_list_overall, Y))

    labels, silhouette_scores = unsupervised_clustering(
        rank_list_overall, return_scores=True)

    print("bcubed.precision", "bcubed.recall", "bcubed.fscore")
    print(evaluate_clustering(Y, labels))

    ns, labels_list = clustering_at_every_n_clusters(rank_list_overall)
    evaluations = np.array([evaluate_clustering(Y, labels)
                            for labels in labels_list])

    n_clusters_found = max(silhouette_scores[0])
    n_clusters_actual = len(np.unique(Y))

    plt.figure(figsize=(6, 4), dpi=200)
    plt.plot(ns, evaluations[:, 0], label="BCubed Precision")
    plt.plot(ns, evaluations[:, 1], label="BCubed Recall")
    plt.plot(ns, evaluations[:, 2], label="BCubed $F_1$ Score")
    plt.plot(*silhouette_scores, label="Silhouette Score")
    plt.axvline(n_clusters_found, 0, 1,
                ls="dashed", c="C3", label="IPS Procedure #Clusters")
    plt.axvline(n_clusters_actual, 0, 1,
                ls="dashed", c="C2", label="Actual #Clusters")
    xmin, xmax, ymin, ymax = plt.axis()
    ypos = ymax / 2 - ymin / 2
    plt.text(n_clusters_found, ypos,
             f"{n_clusters_found}", c="C3", rotation="vertical")
    plt.text(n_clusters_actual, ypos,
             f"{n_clusters_actual}", c="C2", rotation="vertical")
    plt.legend(loc="upper right")
    plt.xlabel("#Clusters")
    plt.ylabel("Metric")
    plt.grid()
    plt.tight_layout()
    plt.savefig("img/unsupervised_clustering.png")


if __name__ == "__main__":
    main()
