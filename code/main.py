from pprint import pprint
import itertools

import matplotlib.pyplot as plt

from corpus import brunet, oxquarry, st_jean
from features import normalize, create_n_grams, mfw
import distances

id, x, y = oxquarry.parse()

X = x
Y = y

# id, x_lemma, x_token, y = brunet.parse()
# X = x_token

X = [[normalize(w) for w in xi] for xi in X]

# to n_grams
X = [create_n_grams(xi, 5) for xi in X]

features = mfw(X, 100)

results = {}
for ai, bi in itertools.combinations(range(len(features)), 2):
    xa = features[ai]
    xb = features[bi]
    results[(ai, bi)] = distances.manhattan(xa, xb)

sorted_results_keys = sorted(results.keys(), key=lambda k: results[k])

for i in sorted_results_keys[0:30]:
    print(Y[i[0]], Y[i[1]], id[i[0]], id[i[1]], results[i])
