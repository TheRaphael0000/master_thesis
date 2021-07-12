"""Compression module

This module provide function that return sizes after compression for document distance based on compression.
This module is mainly used in the linking module in conjonction with the compression distance function in the distances module.
"""

import gzip as gzip_
import bz2 as bz2_
import lzma as lzma_


def gzip(x):
    """The size after using the GZip compression algorithm on the string provided.
    """
    return len(gzip_.compress(x, compresslevel=9))


def bz2(x):
    """The size after using the BZip2 compression algorithm on the string provided.
    """
    return len(bz2_.compress(x, compresslevel=9))


def lzma(x):
    """The size after using the LZMA compression algorithm on the string provided.
    """
    compressor = lzma_.LZMACompressor(preset=9)
    compressor.compress(x)
    compressed = compressor.flush()
    return len(compressed)
