import os
import glob
import re

cwd = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(cwd, "oxquaary")
y_file = "AuthorFile.txt"


def parse():
    x_files = glob.glob(os.path.join(folder,"EX*"))
    id = [re.search(r".*EX(..).txt", f)[1] for f in x_files]

    x = [open(f, "rb").read().decode("utf8").split("\r\n") for f in x_files]
    x = [list(filter("".__ne__, xi)) for xi in x]

    y = open(os.path.join(folder, y_file)).read().split("\n")
    y = [yi for yi in y if yi != ""][:-2]

    return id, x, y


if __name__ == '__main__':
    id, x, y = parse()
    lid = len(id)
    lx = len(x)
    ly = len(y)
    assert lid == lx
    assert lid == ly
    print(f"{lid} texts loaded")
