"""Show the infomations of the datasets."""

from corpus import brunet, oxquarry, st_jean, pan16
from misc import dataset_infos

print("Name Language #Texts #Authors Mean_length #Links r")
id, x, y = oxquarry.parse()
print("Oxquarry EN", *dataset_infos(x, y))
id, x_lemma, x, y = brunet.parse()
print("Brunet FR", *dataset_infos(x, y))
id, x_lemma, x, y = st_jean.parse()
print("St Jean FR", *dataset_infos(x, y))
print("St Jean 1-100 FR", *dataset_infos(x[:100], y[:100]))
print("St Jean 101-200 FR", *dataset_infos(x[100:], y[100:]))


(infos, ids, Xs, Ys) = pan16.parse_train()
for id, x, y in zip(ids, Xs, Ys):
    print(*dataset_infos(x, y))
