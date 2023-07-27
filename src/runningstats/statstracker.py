import numpy as np

from . import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp


class StatsTracker:

    def __init__(self, array_shape, device="cpu"):
        """
        array_shape: shape of the array whose stats we are tracking.
        compute: list of quantities we want to compute (by default all, but can omit some).
        """
        
        # Handle inputs
        self.array_shape = array_shape
        if not ( isinstance(self.array_shape, tuple) or isinstance(self.array_shape, list) ):
            if isinstance(array_shape, int):
                self.array_shape = (self.array_shape,)
            else:
                raise ValueError("array_shape must be list-like or an integer!")

        valid_devices = ["cpu", "gpu"]
        assert device in valid_devices, f"invalid device, must be one of {valid_devices}"
        self.device = device
        if self.device == "cpu":
            self.xp = np
        elif self.device == "gpu" and CUPY_INSTALLED:
            self.xp = cp
        else:
            raise ValueError("Need to install CuPy first if wanting to run on GPU.")

        # Set up
        self.n_samples = 0 # number of samples that have been pushed so far
        self.M1 = self.xp.zeros(self.array_shape)
        self.M2 = self.xp.zeros(self.array_shape)
        self.M3 = self.xp.zeros(self.array_shape)
        self.M4 = self.xp.zeros(self.array_shape)



    def push(self, sample):
        """Updates the running statistics to include the incoming sample.
        """

        # Increment number of samples
        prev_n_samples = self.n_samples
        self.n_samples += 1

        # Update everything else
        delta = sample - self.M1
        delta_n = delta/self.n_samples
        delta_n2 = delta_n*delta_n
        term1 = delta*delta_n*prev_n_samples
        self.M1 += delta_n
        self.M4 += term1*delta_n2*( (self.n_samples**2) - 3*self.n_samples + 3 ) + 6*delta_n2*self.M2 - 4*delta_n*self.M3
        self.M3 += term1*delta_n*(self.n_samples - 2) - 3*delta_n*self.M2
        self.M2 += term1



    def sample_size(self):
        """Returns the sample size.
        """
        return self.n_samples



    def mean(self):
        """Returns the current sample mean.
        """
        return self.M1



    def stdev(self):
        """Returns the current sample standard deviation.
        """
        return np.sqrt(self.variance())



    def variance(self):
        """Returns the current sample variance.
        """
        return self.M2/(self.n_samples - 1.0)



    def skewness(self):
        """Returns the current sample skewness.
        """
        return np.sqrt(self.n_samples)*self.M3/(self.M2**1.5)



    def kurtosis(self):
        """Returns the current sample kurtosis.
        """
        return self.n_samples*self.M4 / (self.M2*self.M2) - 3.0



    def clear(self):
        """Resets the tracker.
        """
        self.n_samples = 0
        self.M1 = self.xp.zeros(self.array_shape)
        self.M2 = self.xp.zeros(self.array_shape)
        self.M3 = self.xp.zeros(self.array_shape)
        self.M4 = self.xp.zeros(self.array_shape)

