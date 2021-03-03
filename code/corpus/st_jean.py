import os
import glob
import re

cwd = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(cwd, "st_jean")
y_file = "Author.txt"


def parse():
    x_files = glob.glob(os.path.join(folder,"CNSaintJean*"))[:100]
    x_lemma = []
    x_token = []

    for f in x_files:
        text = open(f, "rb").read().decode("utf8")
        lines = text.split("\r\n")
        lines = [line for line in lines if line != ""]

        lemma = []
        token = []

        number = False

        for l in lines[2:]:
            m = l.split(",")

            # special cases: comma and dot
            if m == ['"', '"', '"', '"', 'p']:
                m = [",", ",", "p"]
            if m == ['/.', '/.', 'p']:
                m = [".", ".", "P"]

            if 3 != len(m):
                print(f"Error in {f}")
                print(m)
                continue

            if "<Fin nombre>" == m[0]:
                number = False

            s = re.search(r"<Nombre (\d+)>",m[0])
            if s:
                number = True
                token.append(s[1])
                lemma.append(s[1])
                continue

            if number:
                continue

            token.append(m[0])
            lemma.append(m[1])

        assert len(lemma) == len(token)
        x_lemma.append(lemma)
        x_token.append(token)

    id = [re.search(r"CNSaintJean(...).*", f)[1] for f in x_files]
    print(id)

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
