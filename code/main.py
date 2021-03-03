from corpus import brunet, oxquaary, st_jean
from features import normalize, create_n_grams
from collections import Counter
from functools import reduce
from pprint import pprint
import matplotlib.pyplot as plt

# def use_case_oxquaary():
#     id, x, y = oxquaary.parse()
#     print(id)
#
#
# def use_case_brunet():
#     id, x_lemma, x_token, y = brunet.parse()
#     print(id)
#
#
# def use_case_st_jean():
#     id, x_lemma, x_token, y = st_jean.parse()
#     print(id)

# use_case_brunet()
# use_case_oxquaary()
# use_case_st_jean()

id, x_lemma, x_token, y = brunet.parse()

X = x_token

X = [[normalize(w) for w in xi] for xi in X]

# to n_grams
# X = [create_n_grams(xi, 5) for xi in X]

counters = [Counter(xi) for xi in X]
total = reduce(lambda x, y: x + y, counters)
print(total.most_common(100))

# zipf law
distribution = dict(Counter(total.values()))
keys = distribution.keys()
A = list(range(1, 21))
B = [distribution[i] if i in distribution else 0 for i in A]
plt.figure()
plt.xscale("log")
plt.yscale("log")
plt.plot(A,B)
plt.savefig("zipf.png")
