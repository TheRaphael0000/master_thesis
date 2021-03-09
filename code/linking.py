from pprint import pprint
import itertools

from corpus import brunet, oxquarry, st_jean
from features import normalize, create_n_grams, mfw
from misc import zipflaw
import distances
from evaluate import hprec, precision_at_k, rprec, ap

def experiment(X, Y, n_grams, n_mfw, distance_func):
    # Normalization
    X = [[normalize(w) for w in xi] for xi in X]
    # Convert text to n_grams
    if n_grams > 0:
        X = [create_n_grams(xi, n_grams) for xi in X]
    # Create features
    features = mfw(X, n_mfw)
    # Compute link distances
    links_distances = []
    for link in itertools.combinations(range(len(features)), 2):
        ai, bi = link
        xa, xb = features[ai], features[bi]
        links_distances.append((link, distance_func(xa, xb)))

    links_distances.sort(key=lambda x:x[1])

    mesures = ap(links_distances, Y), rprec(links_distances, Y), hprec(links_distances, Y)
    return mesures, links_distances


# Loading dataset
# id, x, y = oxquarry.parse()
id, x_lemma, x_token, y = st_jean.parse()

# Select data
X = x_lemma
Y = y

print(f"AP RPrec HPrec")
print(experiment(X, Y, 5, 500, distances.tanimoto)[0])
