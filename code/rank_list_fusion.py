from collections import defaultdict
from statistics import mean

def fusion(rank_lists, s_curve, args={}):
    # grouping same links
    grouped_by_link = defaultdict(list)
    for rank_list in rank_lists:
        x, y = s_curve(len(rank_list), **args)
        for i, (link, dist) in enumerate(rank_list):
            grouped_by_link[link].append(y[i])
    # average
    for k in grouped_by_link:
        grouped_by_link[k] = mean(grouped_by_link[k])
    overall_ranklist = list(dict(grouped_by_link).items())
    overall_ranklist.sort(key=lambda x: x[-1])
    return overall_ranklist
