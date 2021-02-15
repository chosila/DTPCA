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
exec(open("./info.py").read())

print(year)

class HistCollection(object):
    """Store a collection of cleaned histograms for use in ML algorithms."""

    def __init__(self, hdata, normalize=True, remove_identical_bins=True, extra_info=None, 
                 hist_cleaner=None):
        """
        Initialize the HistCollection.
        
        hdata is a 2D array of histogram data
          Each row is a histogram and each column a bin
        normalize: whether or not to scale histograms to unit area
        remove_identical_bins: remove bins that are the same in every histogram in the collection
        extra_info: dict containing any auxiliary info you want to be stored
          (e.g. extra_info["runs"] could be a list of runs corresponding to each histogram)

        The histograms will be "cleaned" using the HistCleaner class
        """

        self.hdata = np.array(hdata, dtype=float)
        self.__nhists = self.hdata.shape[0]
        self.__nbins = self.hdata.shape[1]
        self.norms = np.sum(hdata, axis=1)

        if hist_cleaner is not None:
            self.__hist_cleaner = hist_cleaner
        else:
            self.__hist_cleaner = HistCleaner(normalize, remove_identical_bins)
        self.__hist_cleaner.fit(self.hdata)
        self.hdata = self.__hist_cleaner.transform(self.hdata)
        

        self.shape = self.hdata.shape
        self.extra_info = extra_info
        
    
    @property
    def nhists(self):
        return self.__nhists

    @property
    def nbins(self):
        return self.__nbins

    @property
    def hist_cleaner(self):
        return self.__hist_cleaner

    @staticmethod
    def draw(h, ax=None, text=None, **kwargs):
        """
        Plot a single histogram with matplotlib.
          - ax: the matplotlib axis to use. Defaults to plt.gca()
          - text: string to write on the plot
          - kwargs: keywork args to pass to pyplot.hist
        """

        if not ax:
            ax = plt.gca()

        if "histtype" not in kwargs:
            kwargs["histtype"] = 'stepfilled'
        if "color" not in kwargs:
            kwargs["color"] = 'k'
        if "linewidth" not in kwargs and "lw" not in kwargs:
            kwargs["lw"] = 1
        if "facecolor" not in kwargs and "fc" not in kwargs:
            kwargs["fc" ] = "lightskyblue"
        if "linestyle" not in kwargs and "ls" not in kwargs:
            kwargs["linestyle"] = '-'

        nbins = h.size
        ax.hist(np.arange(nbins)+0.5, weights=h, bins=np.arange(nbins+1),
                **kwargs)
        ax.set_ylim(0, np.amax(h)*1.5)
        if np.amax(h) > 10000/1.5:
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        if text:
            ax.text(0.05, 0.9, text, transform=ax.transAxes)
        
    def draw_single(self, idx, restore_bad_bins=True, use_normed=False, draw_title=True, **kwargs):
        """
        Plot the histogram at index idx with matplotlib.
          - ax: the matplotlib axis to use. Defaults to plt.gca()
          - restore_bad_bins: use the HistCleaner to restore bins that were removed for plotting
          - use_normed: whether to draw normalized histograms
          - draw_title: whether to draw the title on the plot (extra_info["title"] must exist)
          - kwargs: arguments to pass to the draw function above
        """

        h = self.hdata[idx, :]
        if restore_bad_bins:
            h = self.__hist_cleaner.restore_bad_bins(h)
        if not use_normed:
            h = h*self.norms[idx]

        if draw_title and "title" in self.extra_info:
            kwargs["text"] = self.extra_info["title"]

        HistCollection.draw(h, **kwargs)
