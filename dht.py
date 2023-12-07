import numpy as np

# functions comming from https://github.com/AndreaPi/DHT/tree/master


class DHT():
    def __init__(self):
        self.a = 1

    def dht(self, x):
        """ Compute the DHT for a sequence x of length n using the FFT.
        """
        X = np.fft.fft(x)
        X = np.real(X) - np.imag(X)
        return X

