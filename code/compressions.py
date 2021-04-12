import gzip as gzip_
import zlib as zlib_
import bz2 as bz2_
import lzma as lzma_


def gzip(x):
    return len(gzip_.compress(x))


def zlib(x):
    return len(zlib_.compress(x))


def bz2(x):
    return len(bz2_.compress(x))


def lzma(x):
    return len(lzma_.compress(x))
