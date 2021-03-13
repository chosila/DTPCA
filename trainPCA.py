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

from info import year, plots, norm_cut, max_bins, lumi_json, hpath,HistogramIntegral, getTH1, title
from HistCollection import HistCollection
from DQMPCA import DQMPCA
import pickle

loadPkl = False
PCAnotMade = list()

histzeros = list()
for plot in plots:
    runs = []
    hists = []
    dname = plot[0]
    hname = plot[1]
    fnames = []


    print('-------------------------------------------------------')
    print(dname)
    print(hname)




    if year ==2018:
        fnamesD =glob.glob("/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2018/SingleMuon/*/DQM_V0001_R000*__SingleMuon__Run2018D-PromptReco-v2__DQMIO.root")
        fnamesABC = glob.glob("/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2018/SingleMuon/*/DQM_V0001_R000*__SingleMuon__Run2018*-17Sep2018-*__DQMIO.root")
        fnames= fnamesD +fnamesABC
    
    if year ==2017:
        fnames =glob.glob("/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2017/SingleMuon/*/DQM_V0001_R000*__SingleMuon__Run2017*-17Nov2017-v1__DQMIO.root")
    
    if year ==2016:
        fnames = glob.glob("/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2016/SingleMuon/*/DQM_V0001_R000*__SingleMuon__Run2016*-21Feb2020_UL2016_HIPM-v1__DQMIO.root")
        #fnames = glob.glob("/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2016/SingleMuon/*/DQM_V0001_R000*__SingleMuon__Run2016*-21Feb2020_UL2016_HIPM-v1__DQMIO.root")

    print('# of files in year: ', len(fnames))

    if loadPkl==False:	
        for fname in fnames:
            run = int(fname.split("/")[-1].split("__")[0][-6:])
        
            #Corrupted file
            if run == 315267:
                continue
            
            f = ROOT.TFile.Open(fname)
            
            histograms = getTH1(f, hpath.format(run))
                        
            for name, roothist in histograms:
                
                # ^^ that's because hname is the name of 1 histogram. should we be doing that? 
                if hname in name: #True:
                    #h = roothist.numpy()
                    h = roothist
                            
                    #Include only histograms that have enough events
                    if (norm_cut is None) or (HistogramIntegral(h) >= norm_cut):
                        if max_bins==None:
                            nbins = len(h[0])
                        else:
                            nbins = min(len(h[0]), max_bins)
                        
                        hists.append(h[0])
                        runs.append(run)


        pickle.dump(hists, open('hists.pkl', 'wb'))
    else:
        hists = pickle.load(open('hists.pkl', 'rb'))
    
    print(len(hists))
    if len(hists) < 100:
        print('not enough stats')
        print('num hist passing: ', len(hists))
        PCAnotMade.append(hname)
        continue


    #    print(hists)
    #    if np.count_nonzero(hists) == 0:
    #        histzeros.append(f'{dname}/{hname}')
    #        continue

    try:
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
        
        #print(hists)
        if np.count_nonzero(hists) == 0:
            histzeros.append(f'{dname}/{hname}')
            continue
        
        
        
        ## having troubles when n_good_bins == 0
        hc = HistCollection(hists, extra_info=extra_info)
        
        
        #Fit Modified PCA with Queried Data and Save It
        pca = DQMPCA(norm_cut=norm_cut, sse_ncomps=(1,2,3))
        #print(hc.norms)
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
    except Exception as e: 
        print(traceback.format_exc())
        print(e) 

with open('PCAnotMade.txt', 'w') as filehandle:
    for hist in PCAnotMade:
        filehandle.write(f'{hist}\n')
