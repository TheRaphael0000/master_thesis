from pprint import pprint
import itertools

from corpus import brunet, oxquarry, st_jean
from features import normalize, create_n_grams, mfw
from misc import zipflaw
import distances
from evaluate import hprec, precision_at_k, rprec, ap

# Loading dataset
# id, x, y = oxquarry.parse()
id, x_lemma, x_token, y = st_jean.parse()

# Select data
X = x_token
Y = y

# Normalizationg
X = [[normalize(w) for w in xi] for xi in X]

# Convert text to n_grams
X = [create_n_grams(xi, 5) for xi in X]

# Create features
features = mfw(X, 500)

# Compute link distances
distance_func = distances.manhattan

links_distances = []
for link in itertools.combinations(range(len(features)), 2):
    ai, bi = link
    xa, xb = features[ai], features[bi]
    links_distances.append((link, distance_func(xa, xb)))

links_distances.sort(key=lambda x:x[1])

print("AP", ap(links_distances, Y))
print("RPrec", rprec(links_distances, Y))
print("HPrec", hprec(links_distances, Y))

print("Prec@10", precision_at_k(links_distances, Y, 10))
print("Prec@25", precision_at_k(links_distances, Y, 25))
print("Prec@50", precision_at_k(links_distances, Y, 50))
print("Prec@100", precision_at_k(links_distances, Y, 100))

# for link, dist in links_distances[0:30]:
#     ai, bi = link
#     print(Y[ai], Y[bi], id[ai], id[bi], dist)
