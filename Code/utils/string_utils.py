import numpy as np


def string2bin(text):
    bins = []
    for x in bytearray(text, 'utf8'):
        word = str(format(x, 'b'))
        while len(word) < 8:
            # word = list(word)
            # word.insert(0, '0')
            # word = ''.join(word)
            word = '0' + word
        bins.append(list(word))
    return np.array(bins, dtype=np.int32)


def bin2string(bin_arr):
    byte_string = ''.join(chr(int(bin_arr[i * 8:i * 8 + 8], 2)) for i in range(len(bin_arr) // 8))
    return byte_string
