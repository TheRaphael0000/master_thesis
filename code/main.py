from corpus import brunet
from corpus import oxquaary

def use_case_brunet():
    id, x_lemma, x_token, y = brunet.parse()
    print(x_token[0])

def use_case_oxquaary():
    id, x, y = oxquaary.parse()
    print(x[0])

use_case_brunet()
use_case_oxquaary()
