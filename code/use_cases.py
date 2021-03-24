import numpy as np
from corpus import brunet, oxquarry, st_jean, pan16
import distances
from evaluate import evaluate_linking
from linking import compute_links
import matplotlib.pyplot as plt


def find_best_mfw():
    _, _, X, Y = brunet.parse()
    # _, _, X, Y = st_jean.parse()
    _, X, Y = oxquarry.parse()

    M = []

    print("AP RPrec P@10 HPrec")
    for mfw in np.arange(100, 2500, 200):
        rank_list = compute_links(X, 0, mfw, False, distances.clark)
        mesures = evaluate_linking(rank_list, Y)
        M.append((mfw, *mesures))
        print(mfw, *mesures)


    M = np.array(M)
    fig, axs = plt.subplots(3,1, figsize=(8,8), dpi=200)
    axs[0].plot(M[:,0], M[:,1])
    axs[1].plot(M[:,0], M[:,2])
    axs[2].plot(M[:,0], M[:,4])
    plt.savefig("mfw.png")


if __name__ == '__main__':
    find_best_mfw()
