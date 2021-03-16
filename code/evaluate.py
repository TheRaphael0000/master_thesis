from collections import Counter
import itertools

def rank_list_from_distances_matrix(distances_matrix):
    rank_list = []
    for a, b in itertools.combinations(range(distances_matrix.shape[0]), 2):
        link = a, b
        dist = distances_matrix[link]
        rank_list.append((link, dist))
    rank_list.sort(key=lambda x:x[-1])
    return rank_list

def precision_at_k(rank_list, Y, k):
    sum = 0
    for link, distance in rank_list[0:k]:
        ai, bi = link
        ya, yb = Y[ai], Y[bi]
        if ya == yb:
            sum += 1
    return sum / k


def ap(rank_list, Y):
    sum_precisions = 0
    correct = 0
    for i, (link, distance) in enumerate(rank_list):
        ai, bi = link
        ya, yb = Y[ai], Y[bi]
        if ya == yb:
            correct += 1
            precision_at_i = correct / (i + 1)
            sum_precisions += precision_at_i
    return sum_precisions / correct


def compute_r(Y):
    return int(sum([(n * (n - 1)) / 2 for n in dict(Counter(Y)).values()]))


def rprec(rank_list, Y):
    R = compute_r(Y)
    return precision_at_k(rank_list, Y, R)


def hprec(rank_list, Y):
    i = 0
    for link, distance in rank_list:
        ai, bi = link
        ya, yb = Y[ai], Y[bi]
        if ya == yb:
            i += 1
        else:
            break
    return i
