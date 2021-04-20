import gzip as gzip_
import bz2 as bz2_
import lzma as lzma_


def gzip(x):
    return len(gzip_.compress(x, compresslevel=9))


def bz2(x):
    return len(bz2_.compress(x, compresslevel=9))


def lzma(x):
    compressor = lzma_.LZMACompressor(preset=9)
    compressor.compress(x)
    compressed = compressor.flush()
    return len(compressed)
