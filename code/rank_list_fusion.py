"""Rank list fusion module.

This module focus on fusing rank lists.
"""


import random
from collections import defaultdict

from corpus import brunet
from corpus import oxquarry
from corpus import st_jean
import distances
from evaluate import evaluate_linking
from linking import compute_links
from misc import features_from_rank_list
from misc import labels_from_rank_list
import s_curves

import numpy as np
import scipy as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error


def rank_list_fusion(rls, order=1):
    """Fuse multiple rank list by using the arithmetic mean on their scores

    Arguments:
        rls -- An array of rank lists
        order -- 1 if the small values should be at the top (distance functions), -1 otherwise (default: 1)

    Return:
        rl -- The fused rank list
    """
    # grouping same links
    grp_by_link = defaultdict(list)
    for rl in rls:
        for link, score in rl:
            grp_by_link[link].append(score)
    # averaging
    for k in grp_by_link:
        grp_by_link[k] = np.mean(grp_by_link[k])

    rl = list(dict(grp_by_link).items())
    # shuffle to avoid keeping (score value) in the same order (for debug)
    random.shuffle(rl)
    rl.sort(key=lambda x: order * x[-1])
    return rl


def fusion_z_score(rls):
    """Fuse rank list using the Z-Score strategy

    Arguments:
        rls -- A list of rank lists

    Returns:
        rl -- The fused rank list
    """
    updated_rls = []
    for rl in rls:
        links = [link for link, score in rl]
        scores = np.array([score for link, score in rl])
        infs = scores == np.inf
        ninfs = scores == -np.inf
        scores[infs | ninfs] = np.nan
        zscores = sp.stats.zscore(scores, nan_policy="omit")
        zscores[infs] = np.inf
        zscores[ninfs] = -np.inf
        zscores = list(zscores)
        updated_rls.append(zip(links, zscores))

    return rank_list_fusion(updated_rls)


def fusion_regression_training(rank_list, Y):
    """Train a regression fusion model

    Arguments:
        rank_list -- The rank list to fit the model to (training set)
        Y -- The labels for this rank list (labels)

    Returns:
        model -- The regression logistic model
        float -- The mean squared error on the training set
    """
    Xs = features_from_rank_list(rank_list)
    Ys = labels_from_rank_list(rank_list, Y)
    model = LogisticRegression()
    model.fit(Xs, Ys)
    Y_pred = model.predict_proba(Xs)[:, 1]
    return model, mean_squared_error(Y_pred, Ys, squared=False)


def fusion_regression_trainings(rank_lists, Y):
    """Fit the models for each rank list in the training set

    Arguments:
        rank_lists -- A list of rank lists
        Y -- The labels

    Returns:
        models -- The logistic regression models
        mses -- The mean square errors of the models
    """
    models, mses = [], []
    for rank_list in rank_lists:
        model, mse = fusion_regression_training(rank_list, Y)
        models.append(model)
        mses.append(mse)
    return models, mses


def fusion_regression_testing(model, rank_list):
    """Fit a rank list using a regression logistic trained model

    Arguments:
        model -- The logistic regression trained model
        rank_list -- The rank list to fit the model to

    Returns:
        Y_pred -- The true probability vector for this rank list according to the model
    """
    Xs = features_from_rank_list(rank_list)
    Y_pred = model.predict_proba(Xs)[:, 1]
    return Y_pred


def fusion_regression(models, rls, alter_scores=None):
    """Regression fusion function, takes multiple rank lists and the trained models, return a rank list corresponding to the fusion of the rank list used.

    Arguments:
        models -- The logistic regression trained models
        rls -- The rank lists to fit the models to
        alter_scores -- The function to alter the scores before the fusion, used for the veto strategies

    Returns:
        rl -- A rank list corresponding to the fusion of the provided rank lists according to the fitted models
    """
    updated_rls = []
    for model, rl in zip(models, rls):
        links = [link for link, score in rl]
        scores = fusion_regression_testing(model, rl)
        if alter_scores is not None:
            scores = alter_scores(scores)
        scores = list(scores)
        updated_rls.append(zip(links, scores))
    return rank_list_fusion(updated_rls, order=-1)


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
