import glob
import os
import re

cwd = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(cwd, "brunet")
y_file = "Author.txt"


def parse():
    x_lemma_files = glob.glob(os.path.join(folder, "*lemma*"))
    x_lemma = [open(f, "rb").read().decode("utf8").split("\r\n")
               for f in x_lemma_files]
    # remove empty words
    x_lemma = [list(filter("".__ne__, xi)) for xi in x_lemma]
    # remove repetitions
    x_lemma = [[a for a, b in zip(xi[:-1], xi[1:]) if a != b]
               for xi in x_lemma]

    id = [re.search(r"Tex(..).*", f)[1] for f in x_lemma_files]

    x_token_files = glob.glob(os.path.join(folder, "*token*"))
    x_token = [open(f, "rb").read().decode("utf8").split("\r\n")
               for f in x_token_files]
    x_token = [list(filter("".__ne__, xi)) for xi in x_token]
    x_token = [[a for a, b in zip(xi[:-1], xi[1:]) if a != b]
               for xi in x_token]

    y = open(os.path.join(folder, y_file)).read().split("\n")
    y = [yi for yi in y if yi != ""]

    return id, x_lemma, x_token, y


if __name__ == '__main__':
    id, x_lemma, x_token, y = parse()
    lid = len(id)
    lx_lemma = len(x_lemma)
    lx_token = len(x_token)
    ly = len(y)
    assert lid == lx_lemma
    assert lid == lx_token
    assert lid == ly
    print(f"{lid} texts loaded")
