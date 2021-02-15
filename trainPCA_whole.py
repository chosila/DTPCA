###################################
#
# this is just the other classes and training script but put into 1 file 
# this looks likethe jupyter notebook, but might be harder to read? 
#
##################################
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

#Define variables for bulk training
year = 2016
norm_cut = None #10000 ## changing normcut for now because nothing is passing
max_bins = None
title = None
lumi_json = None
hpath = "DQMData/Run {}/DT/Run summary"
plots = [("02-Segments/Wheel0/Sector1/Station1","T0_FromSegm_W0_Sec1_St1")]#,("Station2","T0_FromSegm_W0_Sec1_St2"),("Station3","T0_FromSegm_W0_Sec1_St3"),("Station4","T0_FromSegm_W0_Sec1_St4")]

#HistogramIntegral returns the total number of events
def HistogramIntegral(hist):
    return sum(hist[0][i] for i in range(len(hist[0])))


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

class DQMPCA(object):
    """Class to perform PCA specifically on HistCollection objects"""

    def __init__(self, use_standard_scaler=False, norm_cut=norm_cut, sse_ncomps=None):
        """Initialize the DQMPCA

        -use_standard_scalar determines whether to use standard scaling
          (zero mean, unit stdev) before feeding into a PCA. This helps
          for some histograms, but hurts for others
        """
        if use_standard_scaler:
            self.pca = Pipeline(
                ("scaler", StandardScaler()),
                ("pca", PCA())
                )
        else:
            self.pca = PCA()

        self.use_standard_scaler = use_standard_scaler
        self.norm_cut = norm_cut
        self.sse_ncomps = sse_ncomps

        self.__is_fit = False

    @property
    def sse_ncomps(self):
        return self.__sse_ncomps

    @sse_ncomps.setter
    def sse_ncomps(self, sse):
        if sse is not None and not isinstance(sse, tuple) and not isinstance(sse, list):
            raise Exception("illigal sse_ncomps value. Should be None or a list/tuple of ints")
        self.__sse_ncomps = sse

    def _check_fit(self):
        if not self.__is_fit:
            raise Exception("Must fit the DQMPCA before calling transform")

    def fit(self, hdata):
        if isinstance(hdata, HistCollection):
            self._hist_cleaner = hdata.hist_cleaner
            cleaned = hdata.hdata
            norms = hdata.norms
            
        else:
            self._hist_cleaner = HistCleaner()
            self._hist_cleaner.fit(hdata)
            cleaned = self._hist_cleaner.transform(hdata)
            norms = np.sum(cleaned, axis=1)

        cleaned = cleaned[norms>self.norm_cut, :]
        self.pca.fit(cleaned)        
        self.__is_fit = True

        if self.sse_ncomps is not None:
            self.sse_cuts = {}
            for ncomp in self.sse_ncomps:
                self.sse_cuts[ncomp] = []
                sses = self.sse(cleaned, ncomp)
                for pct in np.arange(1,101):
                    self.sse_cuts[ncomp].append(np.percentile(sses, pct))
    
    def transform(self, hdata):
        """Transform a set of histograms with the trained PCA"""
        self._check_fit()
        if isinstance(hdata, HistCollection):
            cleaned = hdata.hdata
        else:
            cleaned = self._hist_cleaner.transform(hdata)        
        return self.pca.transform(cleaned)
        
    def inverse_transform(self, xf, n_components=3, restore_bad_bins=False):
        self._check_fit()
        xf = np.array(xf)
        trunc = np.zeros((xf.shape[0], self._hist_cleaner.n_good_bins))
        trunc[:,:n_components] = xf[:,:n_components]

        #ixf = self.pca.inverse_transform(trunc)
        ## making it a transpose seems to fix the dimension mismatch
        ixf = self.pca.inverse_transform(trunc.transpose())
        
        if not restore_bad_bins:
            return ixf
        else:
            return self._hist_cleaner.restore_bad_bins(ixf)

    def sse(self, hdata, n_components=3):
        if isinstance(hdata, HistCollection):
            cleaned = hdata.hdata
        else:
            cleaned = self._hist_cleaner.transform(hdata)        
        xf = self.transform(cleaned)
        ixf = self.inverse_transform(xf, n_components=n_components)
        
        
        return np.sqrt(np.sum((ixf-cleaned)**2, axis=1))
        
    def score(self, hdata, n_components=3):
        if not hasattr(self, "sse_cuts") or n_components not in self.sse_cuts:
            raise Exception("must fit first with {0} in sse_ncomps".format(n_components))
        sse = self.sse(hdata, n_components)
        return np.interp(sse, self.sse_cuts[n_components], np.arange(1,101))

    @property
    def explained_variance_ratio(self):
        if self.use_standard_scaler:
            return self.pca.named_steps["pca"].explained_variance_ratio_
        else:
            return self.pca.explained_variance_ratio_

    @property
    def mean(self):
        if self.use_standard_scaler:
            return self.pca.named_steps["scaler"].inverse_transform(self.pca.named_steps["pca"].mean_)
        else:
            return self.pca.mean_


## loops through directory in root file and returns the histogram path and a numpy 
## of the histogram
def getTH1(d, h):
    keyName = d.GetDirectory(h)
    for key in keyName.GetListOfKeys():
        obj = keyName.Get(key.GetName())
        ## this is so we don't return a null pointer
        if obj:
            objName = obj.GetName()
            if obj.IsFolder():
                for i in getTH1(d, h+'/'+objName):
                    yield i
            else:
                if obj.InheritsFrom('TH1'):
                    yield (h+objName, root_numpy.hist2array(obj, return_edges=True))
        else: 
            
            continue

                
for plot in plots:
    runs = []
    hists = []
    dname = plot[0]
    hname = plot[1]
    fnames = []


    if year ==2018:
        fnamesD =glob.glob("/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2018/SingleMuon/*/DQM_V0001_R000*__SingleMuon__Run2018D-PromptReco-v2__DQMIO.root")
        fnamesABC = glob.glob("/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2018/SingleMuon/*/DQM_V0001_R000*__SingleMuon__Run2018*-17Sep2018-*__DQMIO.root")
        fnames= fnamesD +fnamesABC
    
    if year ==2017:
        fnames =glob.glob("/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2017/SingleMuon/*/DQM_V0001_R000*__SingleMuon__Run2017*-17Nov2017-v1__DQMIO.root")
    
    if year ==2016:
        fnames = glob.glob("/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2016/SingleMuon/*/DQM_V0001_R000*__SingleMuon__Run2016*-21Feb2020_UL2016_HIPM-v1__DQMIO.root")
    
    for fname in fnames:
        run = int(fname.split("/")[-1].split("__")[0][-6:])

        #Corrupted file
        if run == 315267:
            continue
        #if run != 272760:
        #    continue
        #f = uproot.open(fname)
        #Fetch all the 1D histograms into a list
        
        ## i'm keeping the number of runs short so i can debug fast
        if run < 275000:#272776:
            continue
        if run > 275778:
            break
        
        f = ROOT.TFile.Open(fname)
        
        histograms = getTH1(f, hpath.format(run))
        
        #histograms =f[hpath.format(run)].allitems(filterclass=lambda cls: issubclass(cls, uproot_methods.classes.TH1.Methods))
        print(run)
        
        
        
        for name, roothist in histograms:
            
            #name = name.decode("utf-8")
            #name = name.replace(";1", "")
            #Grab the 1D histogram we want
            #if (dname in name) and (hname in name): 
            
            # when use hname in name, only 1 histogram passes the cut. should that be the case?
            # ^^ that's because hname is the name of 1 histogram. should we be doing that? 
            if True:#hname in name: #True:
                #h = roothist.numpy()
                h = roothist

                #Include only histograms that have enough events
                if norm_cut is None or HistogramIntegral(h) >= norm_cut:
                    if max_bins==None:
                        nbins = len(h[0])
                    else:
                        nbins = min(len(h[0]), max_bins)

                    hists.append(h[0])
                    runs.append(run)
    print(hists[0])

    #Make rows even length if jagged
    lens = [len(row) for row in hists]
    maxlen = np.amax(lens)
    if maxlen != np.amin(lens):
        for i in range(len(hists)):
            hists[i] = np.ndarray.tolist(hists[i])
    hists = np.array(hists)
    #Define extra infos such as run number, title, and luminosity
    #To be updated: query lumi data from OMS
    extra_info = {"runs":runs}
    extra_info["title"]  = title
    if lumi_json is not None:
        with open(lumi_json, 'rb') as fid:
            ri = json.load(fid)
        lumis = []
        for run in runs:
            if str(run) in ri:
                A = ri[str(run)]["Initial Lumi"]
                B = ri[str(run)]["Ending Lumi"]
                if A<0.1 or B<0.1:
                    lumis.append(0)
                elif A==B:
                    lumis.append(A)
                else:
                    lumis.append((A-B)/np.log(A/B))
            else:
                lumis.append(0)
        extra_info["lumis"] = np.array(lumis)
    #Clean and Store Histogram Data Using Class Object
    
    ## having troubles when n_good_bins == 0
    hc = HistCollection(hists, extra_info=extra_info)
    
    
    #Fit Modified PCA with Queried Data and Save It
    pca = DQMPCA(norm_cut=10000, sse_ncomps=(1,2,3))
    pca.fit(hc)
    pkl_dir = os.path.join("DT_models",str(year))
    os.system("mkdir -p "+pkl_dir)
    pkl_filename = "{0}_{1}.pkl".format(dname, hname)
    pkl_filename = os.path.join(pkl_dir,pkl_filename)
    with open(pkl_filename, 'wb') as file:
        #Protocol can be set higher once the code is moved to Python 3
        pickle.dump(pca, file,protocol = 2)
