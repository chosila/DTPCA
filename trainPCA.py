####################################
#
# script using HistCollection, DQMPCA to train
#
####################################

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
#exec(open('./HistCleaner.py').read())
#exec(open('./DQMPCA.py').read())
from info import year, plots, norm_cut, max_bins, lumi_json, hpath,HistogramIntegral, getTH1, title
from HistCollection import HistCollection
from DQMPCA import DQMPCA


for plot in plots:
    runs = []
    hists = []
    dname = plot[0]
    hname = plot[1]
    fnames = []


    print(dname)
    print(hname)




    if year ==2018:
        fnamesD =glob.glob("/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2018/SingleMuon/*/DQM_V0001_R000*__SingleMuon__Run2018D-PromptReco-v2__DQMIO.root")
        fnamesABC = glob.glob("/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2018/SingleMuon/*/DQM_V0001_R000*__SingleMuon__Run2018*-17Sep2018-*__DQMIO.root")
        fnames= fnamesD +fnamesABC
    
    if year ==2017:
        fnames =glob.glob("/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2017/SingleMuon/*/DQM_V0001_R000*__SingleMuon__Run2017*-17Nov2017-v1__DQMIO.root")
    
    if year ==2016:
        fnames = glob.glob("/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2016/SingleMuon/*/DQM_V0001_R000*__SingleMuon__Run2016F*-21Feb2020_UL2016_HIPM-v1__DQMIO.root")
        #fnames = glob.glob("/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2016/SingleMuon/*/DQM_V0001_R000*__SingleMuon__Run2016*-21Feb2020_UL2016_HIPM-v1__DQMIO.root")
    
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
        #if run < 275000:#272776:
        #    continue
        #if run > 275778:
        #    break
        
        f = ROOT.TFile.Open(fname)
        
        histograms = getTH1(f, hpath.format(run))
        
        #histograms =f[hpath.format(run)].allitems(filterclass=lambda cls: issubclass(cls, uproot_methods.classes.TH1.Methods))
        #print(run)
        
        
        
        for name, roothist in histograms:
            
            #name = name.decode("utf-8")
            #name = name.replace(";1", "")
            #Grab the 1D histogram we want
            #if (dname in name) and (hname in name): 
            
            # when use hname in name, only 1 histogram passes the cut. should that be the case?
            # ^^ that's because hname is the name of 1 histogram. should we be doing that? 
            if hname in name: #True:
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
    #os.system("mkdir -p "+pkl_dir)
    #pkl_filename = "{0}_{1}.pkl".format(dname, hname)
    pkl_filename = f'{hname}.pkl'
    pkl_filename = os.path.join(pkl_dir,pkl_filename)
    os.makedirs(os.path.dirname(pkl_filename), exist_ok=True)
    with open(pkl_filename, 'wb') as file:
        #Protocol can be set higher once the code is moved to Python 3
        pickle.dump(pca, file,protocol = 2)
