from pprint import pprint

from corpus import brunet, oxquarry, st_jean
from evaluate import compute_r
from statistics import mean

print(f"Name, Language, #Texts, #Authors, Mean length, #Links")
id, x, y = oxquarry.parse()
print(f"Oxquarry, EN, {len(y)}, {len(set(y))}, {mean([len(xi) for xi in x]):.2f}, {compute_r(y)}")
id, x_lemma, x, y = brunet.parse()
print(f"Brunet, FR, {len(y)}, {len(set(y))}, {mean([len(xi) for xi in x]):.2f}, {compute_r(y)}")
id, x_lemma, x, y = st_jean.parse()
print(f"St Jean, FR, {len(y)}, {len(set(y))}, {mean([len(xi) for xi in x]):.2f}, {compute_r(y)}")
