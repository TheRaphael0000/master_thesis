import os
import glob
import re
import string
import json

cwd = os.path.dirname(os.path.abspath(__file__))

dataset_path = "pan16"

path = os.path.join(cwd, dataset_path)
train_x_path = os.path.join(path, "train")
train_y_path = os.path.join(train_x_path, "truth")

test_x_path = os.path.join(path, "test")
test_y_path = os.path.join(test_x_path, "truth")

y_filename = "clustering.json"


def parse(x_path, y_path):
    info = json.load(open(os.path.join(x_path, "info.json")))
    x_paths = glob.glob(os.path.join(x_path, "problem*"))
    y_paths = glob.glob(os.path.join(y_path, "problem*"))

    ids = []
    xs = []
    ys = []
    for x_path, y_path in zip(x_paths, y_paths):
        documents_paths = glob.glob(os.path.join(x_path, "document*"))
        clustering_file = os.path.join(y_path, y_filename)
        id, x, y = parse_problem(documents_paths, clustering_file)
        ids.append(id)
        xs.append(x)
        ys.append(y)

    return list(zip(info, ids, xs, ys))


def parse_problem(documents_paths, clustering_file):
    id = [os.path.basename(d) for d in documents_paths]
    x = [parse_file(p) for p in documents_paths]
    js = json.load(open(clustering_file))
    cluster_lookup = {}
    for i, cluster in enumerate(js):
        documents = [d["document"] for d in cluster]
        for document in documents:
            cluster_lookup[document] = i
    y = []
    for idi in id:
        y.append(cluster_lookup[idi])
    return id, x, y


def parse_file(file_path):
    txt = open(file_path, "rb").read().decode("utf8")
    txt = txt.lower()
    txt = "".join([l for l in txt if l in string.printable])
    for a in string.punctuation:
        txt = txt.replace(a, f" {a} ")
    tokens = txt.split(" ")
    tokens = [t for t in tokens if t != ""]
    return tokens


def parse_train():
    return parse(train_x_path, train_y_path)


def parse_test():
    return parse(test_x_path, test_y_path)


def test(func):
    problems = func()

    for info, id, x, y in problems:
        lid = len(id)
        lx = len(x)
        ly = len(y)
        assert lid == lx
        assert lid == ly
        print(f"{info} {lid} texts loaded")


if __name__ == '__main__':
    print("Testing parse_train...")
    test(parse_train)
    print("Testing parse_test...")
    test(parse_test)
