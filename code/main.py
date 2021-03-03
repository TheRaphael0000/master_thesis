from corpus import brunet
from corpus import oxquaary
from corpus import st_jean

def use_case_oxquaary():
    id, x, y = oxquaary.parse()
    print(id)

def use_case_brunet():
    id, x_lemma, x_token, y = brunet.parse()
    print(id)

def use_case_st_jean():
    id, x_lemma, x_token, y = st_jean.parse()
    print(id)

use_case_brunet()
use_case_oxquaary()
use_case_st_jean()
