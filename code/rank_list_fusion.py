"""Rank list fusion module."""


from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy as sp

from sklearn.metrics import mean_squared_error

import s_curves
import distances
from corpus import brunet, oxquarry, st_jean
from linking import compute_links
from collections import defaultdict
from evaluate import evaluate_linking
from misc import features_from_rank_list
from misc import labels_from_rank_list


def rank_list_fusion(links, scores, order=1):
    n_lists = links.shape[0]
    n_links = links.shape[1]

    # grouping same links
    grp_by_link = defaultdict(list)
    for list_indice in range(n_lists):
        for link_indice in range(n_links):
            link = links[list_indice, link_indice]
            score = scores[list_indice, link_indice]
            grp_by_link[tuple(link)].append(score)
    # averaging
    for k in grp_by_link:
        grp_by_link[k] = np.mean(grp_by_link[k])
    rl = list(dict(grp_by_link).items())
    rl.sort(key=lambda x: order*x[-1])
    return rl


def fusion_z_score(rank_lists):
    links = np.array([[link for link, score in rl] for rl in rank_lists])
    scores = np.array([[score for link, score in rl] for rl in rank_lists])
    zscores = sp.stats.zscore(scores, axis=1)
    return rank_list_fusion(links, zscores)


def fusion_regression_training(rank_list, Y):
    Xs = features_from_rank_list(rank_list)
    Ys = labels_from_rank_list(rank_list, Y)
    model = LogisticRegression()
    model.fit(Xs, Ys)
    Y_pred = model.predict_proba(Xs)[:,1]
    return model, mean_squared_error(Y_pred, Ys, squared=False)


def fusion_regression_testing(model, rank_list):
    Xs = features_from_rank_list(rank_list)
    Y_pred = model.predict_proba(Xs)
    return Y_pred


def fusion_regression(models, rank_lists):
    links = np.array([[link for link, score in rl] for rl in rank_lists])
    scores = np.array([fusion_regression_testing(model, rl) for model, rl in zip(models, rank_lists)])
    return rank_list_fusion(links, scores, order=-1)


if __name__ == '__main__':
    _, X1, Y1 = oxquarry.parse()
    _, _, X2, Y2 = brunet.parse()
    _, _, _, X3, Y3 = st_jean.parse_A()
    _, _, _, X4, Y4 = st_jean.parse_B()

    def experiments_(X):
        return [
            [X, 0, 500, True, 0.1, distances.manhattan],
            [X, 0, 500, True, 0.1, distances.cosine_distance]
        ]

    X_training, Y_training = X2, Y2
    X_testing, Y_testing = X4, Y4

    rls = [compute_links(e) for e in experiments_(X_testing)]
    s_curve = s_curves.sigmoid_reciprocal(c=4, r=0.1)
    rls = [s_curves.soft_veto(rl, s_curve) for rl in rls]
    print("Rank lists")
    for rl in rls:
        print(*evaluate_linking(rl, Y_testing))

    print("Z-score")
    rl_overall = fusion_z_score(rls)
    print(*evaluate_linking(rl_overall, Y_testing))

    print("Regression")
    print("Train")
    rls_training = [compute_links(e) for e in experiments_(X_training)]
    models = []
    for rl in rls_training:
        model, rmse = fusion_regression_training(rl, Y_training)
        models.append(model)
        print(*evaluate_linking(rl, Y_training), rmse)
    print("Test")
    rl_overall = fusion_regression(models, rls)
    print(*evaluate_linking(rl_overall, Y_testing))
