import os
import glob

cwd = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(cwd, "brunet")
y_file = "Author.txt"


def parse():
    x_lemma_files = glob.glob(folder+"/*lemma*")
    x_lemma = [open(f, "rb").read().decode("utf8").split("\r\n") for f in x_lemma_files]
    x_lemma = [list(filter("".__ne__, xi)) for xi in x_lemma]

    x_token_files = glob.glob(folder+"/*token*")
    x_token = [open(f, "rb").read().decode("utf8").split("\r\n") for f in x_token_files]
    x_token = [list(filter("".__ne__, xi)) for xi in x_token]

    y = open(os.path.join(folder, y_file)).read().split("\n")
    y = [yi for yi in y if yi != ""]

    return x_lemma, x_token, y


if __name__ == '__main__':
    x_lemma, x_token, y = parse()
    lx_lemma = len(x_lemma)
    lx_token = len(x_token)
    ly = len(y)
    assert lx_token == lx_lemma
    assert ly == lx_token
    print(f"{ly} texts loaded")
