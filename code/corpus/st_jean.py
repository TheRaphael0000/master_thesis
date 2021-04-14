import os
import glob
import re

cwd = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(cwd, "st_jean")
y_file = os.path.join(folder, "Author.txt")
info_file = os.path.join(folder, "infos.csv")

def parse_A():
    return parse_(0, 100)

def parse_B():
    return parse_(100, 100)

def parse():
    return parse_(0, 200)

def parse_(offset, size):
    x_files = glob.glob(os.path.join(folder, "CNSaintJean*"))[offset:offset+size]
    x_lemma = []
    x_token = []
    x_pos = []

    for f in x_files:
        text = open(f, "rb").read().decode("utf8")
        lines = text.split("\r\n")
        lines = [line for line in lines if line != ""]

        lemma = []
        token = []
        pos = []

        number = False

        for l in lines[2:]:
            m = l.split(",")

            # special cases: comma and dot
            if m == ['"', '"', '"', '"', 'p']:
                m = [",", ",", "p"]
            if m[-1] == "P":
                m = [w.replace("/", "") for w in m]
            # ignore double "des"
            if m == ["des", "le", "71"]:
                continue

            if 3 != len(m):
                raise Exception(f"Error in {f}\n {m} is not a 3 value field")
                continue

            if "<Fin nombre>" == m[0]:
                number = False

            s = re.search(r"<Nombre (\d+)>", m[0])
            if s:
                number = True
                token.append(s[1])
                lemma.append(s[1])
                pos.append(72)
                continue

            if number:
                continue

            token.append(m[0])
            lemma.append(m[1])
            try:
                pos.append(int(m[2]))
            except ValueError:
                pos.append(100 if m[2] == 'p' else 101)

        assert len(lemma) == len(token)
        assert len(lemma) == len(pos)
        x_lemma.append(lemma)
        x_token.append(token)
        x_pos.append(pos)

    # id = [re.search(r"CNSaintJean(...).*", f)[1] for f in x_files][offset:offset+size]
    infos = [l.split(",") for l in open(info_file, "rb").read().decode("utf8").split("\r\n")[1:-1]][offset:offset+size]

    y = open(y_file).read().split("\n")
    y = [yi for yi in y if yi != ""][offset:offset+size]

    return infos, x_pos, x_lemma, x_token, y


def print_texts(data):
    infos, x_pos, x_lemma, x_token, y = data
    for line in infos:
        def f(x):
            author_text = x[2].split(" - ")
            return f"{x[0]} & {author_text[0]} & {author_text[-1]} & {x[-1]}"
        print(f"{f(line)} \\\\")
        # print(infos)


if __name__ == '__main__':
    data = parse()
    print_texts(data)
    infos, x_pos, x_lemma, x_token, y = data
    linfos = len(infos)
    lx_lemma = len(x_lemma)
    lx_token = len(x_token)
    lx_pos = len(x_pos)
    ly = len(y)
    assert linfos == lx_lemma
    assert linfos == lx_token
    assert linfos == lx_pos
    assert linfos == ly
    print(f"{linfos} texts loaded")
