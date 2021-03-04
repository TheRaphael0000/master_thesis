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
    sum = 0
    correct = 0
    for i, (link, distance) in enumerate(links_distances):
        ai, bi = link
        ya, yb = Y[ai], Y[bi]
        if ya == yb:
            correct += 1
            sum += correct / (i + 1)
    return sum / len(links_distances)


def rprec(links_distances, Y):
    # find R
    R = int(sum([(n * (n - 1)) / 2 for n in dict(Counter(Y)).values()]))
    print(R)
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
