import torch
import numpy as np
from PIL import Image
from text import texts
from utils.string_utils import string2bin, bin2string


class BaseStego:
    DELIMITER = np.ones(100, dtype=int)    # TODO hidden info ends with 1, then decoder skip it

    def __init__(self):
        pass

    @staticmethod
    def encode(container, information):
        raise NotImplementedError

    @staticmethod
    def decode(container):
        raise NotImplementedError


class LSBMatching(BaseStego):
    def __init__(self):
        super(LSBMatching, self).__init__()

    @staticmethod
    def encode(container, information, stego='stego.png'):
        """
        LSB Matching algorithm (+-1 embedding)
        :param container: path to image container
        :param information: array with int bits
        :param stego: name of image with hidden message
        """
        img = Image.open(container)
        width, height = img.size
        img_matr = np.array(img)
        img_matr.setflags(write=True)

        red_chan = img_matr[:, :, 0].reshape((1, -1))[0]

        information = np.append(information, BaseStego.DELIMITER)   # add end signal

        for i, bit in enumerate(information):
            if bit != red_chan[i] & 1:
                if np.random.randint(0, 2) == 0:
                    red_chan[i] -= 1
                else:
                    red_chan[i] += 1

        img_matr[:, :, 0] = red_chan.reshape((height, width))
        Image.fromarray(img_matr).save(stego)

    @staticmethod
    def decode(container):
        img = Image.open(container)
        img_matr = np.asarray(img)

        red_chan = img_matr[:, :, 0].reshape((1, -1))[0]
        delim_len = len(BaseStego.DELIMITER)

        info = np.array([], dtype=int)
        for pixel in red_chan:
            info = np.append(info, [pixel & 1])

            if info.shape[0] > delim_len and np.array_equiv(info[-delim_len:], BaseStego.DELIMITER):
                break

        info = info[:-delim_len]

        stego = ''.join(map(str, info))
        while len(stego) % 8 != 0:
            stego += '1'
        return stego
