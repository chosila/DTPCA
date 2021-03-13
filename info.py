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
year = 2018
norm_cut = 10000 ## changing normcut for now because nothing is passing
max_bins = None
title = None
lumi_json = None
hpath = "DQMData/Run {}/DT/Run summary"
#plots = [("02-Segments/Wheel0/Sector1/Station1","T0_FromSegm_W0_Sec1_St1")]
plots=list() #[('02-Segments/Wheel0/Sector13/Station1', 'T0_FromSegm_W0_Sec13_St1')]

for Sec in range(1,15):
    for St in [1,2,3,4]:
        plots.append((f'02-Segments/Wheel0/Sector{Sec}/Station{St}', f'T0_FromSegm_W0_Sec{Sec}_St{St}'))

for Sec in range(1,7):
    for St in [1,2,3,4]:
        plots.append((f'02-Segments/Wheel1/Sector{Sec}/Station{St}', f'T0_FromSegm_W1_Sec{Sec}_St{St}'))



## define what plots to train
## each tuple is (dirname, histname)

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
                    yield (h+'/'+objName, root_numpy.hist2array(obj, return_edges=True))
        else: 
            
            continue
