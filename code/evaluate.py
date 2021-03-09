from collections import Counter


def precision_at_k(links_distances, Y, k):
    sum = 0
    for link, distance in links_distances[0:k]:
        ai, bi = link
        ya, yb = Y[ai], Y[bi]
        if ya == yb:
            sum += 1
    return sum / k


def ap(links_distances, Y):
    sum_precisions = 0
    correct = 0
    for i, (link, distance) in enumerate(links_distances):
        ai, bi = link
        ya, yb = Y[ai], Y[bi]
        if ya == yb:
            correct += 1
            precision_at_i = correct / (i + 1)
            sum_precisions += precision_at_i
    return sum_precisions / correct


def compute_r(Y):
    return int(sum([(n * (n - 1)) / 2 for n in dict(Counter(Y)).values()]))


def rprec(links_distances, Y):
    R = compute_r(Y)
    return precision_at_k(links_distances, Y, R)


def hprec(links_distances, Y):
    i = 0
    for link, distance in links_distances:
        ai, bi = link
        ya, yb = Y[ai], Y[bi]
        if ya == yb:
            i += 1
        else:
            break
    return i
