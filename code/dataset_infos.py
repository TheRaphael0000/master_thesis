from pprint import pprint

from corpus import brunet, oxquarry, st_jean
from evaluate import dataset_infos

print(f"Name Language #Texts #Authors Mean_length #Links")
id, x, y = oxquarry.parse()
print("Oxquarry EN", *dataset_infos(x, y))
id, x_lemma, x, y = brunet.parse()
print("Brunet FR", *dataset_infos(x, y))
id, x_lemma, x, y = st_jean.parse()
print("St Jean FR", *dataset_infos(x, y))
