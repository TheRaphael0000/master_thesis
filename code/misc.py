import matplotlib.pyplot as plt

def zipflaw(total, n=21):
    distribution = dict(Counter(total.values()))
    keys = distribution.keys()
    A = list(range(1, n))
    B = [distribution[i] if i in distribution else 0 for i in A]
    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(A, B)
    plt.savefig("zipf.png")
