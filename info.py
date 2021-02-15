###############################
#
# define training related variables as well as other functions 
# needed to train
#
##############################


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
year = 2017
norm_cut = None #10000 ## changing normcut for now because nothing is passing
max_bins = None
title = None
lumi_json = None
hpath = "DQMData/Run {}/DT/Run summary"
plots = [("02-Segments/Wheel0/Sector1/Station1","T0_FromSegm_W0_Sec1_St1")]#,("Station2","T0_FromSegm_W0_Sec1_St2"),("Station3","T0_FromSegm_W0_Sec1_St3"),("Station4","T0_FromSegm_W0_Sec1_St4")]

#HistogramIntegral returns the total number of events
def HistogramIntegral(hist):
    return sum(hist[0][i] for i in range(len(hist[0])))


## loops through directory in root file and returns the histogram path and a numpy 
## of the histogram. This replaces uproot.allItems() as DT plots struggle with uproot
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
