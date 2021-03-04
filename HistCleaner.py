import numpy as np
import pandas as pd
import os
import ROOT
import uproot
import uproot_methods
import pickle
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import glob
import matplotlib.pyplot as plt
import root_numpy

## execute needed files
#exec(open("./info.py").read())
#exec(open("./HistCollection.py").read())



class HistCleaner(object):
    """ 
    sklearn-style preprocessing class to perform necessary "cleaning" of histogram collections for use in ML algorithms
    
    Can perform two separate operations, controlled by boolean flags:
     - normalize: whether or not to scale histograms to unit area
     - remove_identical_bins: remove bins that are the same in every histogram in the collection
    """
    def __init__(self, normalize=True, remove_identical_bins=True):
        self.__normalize = normalize
        self.__remove_identical_bins = remove_identical_bins

        # internal use
        self.__is_fit = False

    
    @property
    def normalize(self):
        return self.__normalize

    @normalize.setter
    def normalize(self, norm):
        if not isinstance(norm, bool):
            raise Exception("normalize must be set to a boolean value")
        self.__normalize = norm

    @property
    def remove_identical_bins(self):
        return self.__remove_identical_bins

    @remove_identical_bins.setter
    def remove_identical_bins(self, rib):
        if not isinstance(rib, bool):
            raise Exception("remove_identical_bins must be set to a boolean value")
        self.__remove_identical_bins = rib

    def fit(self, hd):
        self.nbins = hd.shape[1]
        # find the "good" bin indices (those that aren't the same in every histogram)
        #np.tile transform and repeat a given array
        bad_bins = np.all(hd==np.tile(hd[0,:],hd.shape[0]).reshape(hd.shape), axis=0)
        
        good_bins = np.logical_not(bad_bins)
        self.bad_bins = np.arange(self.nbins)[bad_bins]
        self.good_bins = np.arange(self.nbins)[good_bins]
        self.n_good_bins = self.good_bins.size
        self.bad_bin_contents = hd[0,self.bad_bins]

        self.__is_fit = True

    def _check_fit(self):
        if not self.__is_fit:
            raise Exception("Must fit the HistCleaner before calling transform")

    def restore_bad_bins(self, hd):
        self._check_fit()
        init_shape = hd.shape
        if len(init_shape) == 1:
            hd = hd.reshape(1,-1)
        if hd.shape[1] != self.n_good_bins:
            raise Exception("Invalid number of columns")

        ret = np.zeros((hd.shape[0], self.nbins))
        ret[:,self.good_bins] = hd
        ret[:,self.bad_bins] = np.tile(self.bad_bin_contents, hd.shape[0]).reshape(hd.shape[0], self.bad_bins.size)

        if len(init_shape) == 1:
            ret = ret.reshape(ret.size,)
        return ret

    def remove_bad_bins(self, hd):
        self._check_fit() 
        init_shape = hd.shape
        if len(init_shape) == 1:
            hd = hd.reshape(1,-1)
        if hd.shape[1] != self.nbins:
            raise Exception("Invalid number of columns")
        
        ret = hd[:,self.good_bins]
        if len(init_shape) == 1:
            ret = ret.reshape(ret.size,)
        return ret

    def transform(self, hd):
        self._check_fit()
        init_shape = hd.shape
        if len(init_shape)==1:
            hd = hd.reshape(1,-1)
        is_cleaned = False
        if hd.shape[1] != self.nbins:
            if hd.shape[1] == self.n_good_bins:
                is_cleaned = True
            else:
                raise Exception("Invalid shape! Expected {0} or {1} columns, got {2}".format(self.nbins,self.n_good_bins, hd.shape[1]))

        # remove bad bins
        if not is_cleaned and self.remove_identical_bins:
            hd = self.remove_bad_bins(hd)

        # normalize each row
        if self.normalize:
            norms = np.sum(hd, axis=1)
            tile = np.tile(norms, self.n_good_bins).reshape(self.n_good_bins, -1).T
            hd = np.divide(hd, tile, out=np.zeros_like(hd), where=tile!=0)

        if len(init_shape) == 1:
            hd = hd.reshape(hd.size,)
        return hd

    def fit_transform(self, hd):
        self.fit(hd)
        return self.transform(hd)
