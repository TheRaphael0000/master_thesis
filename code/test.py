def test():
    for a in range(10):
        yield a
    yield -1


for a in test():
    print(a)
