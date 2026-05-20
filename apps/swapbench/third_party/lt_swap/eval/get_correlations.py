
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os,sys,json
import numpy as np
from scipy import stats

def get_score(res_dir,tag):
    lastbin,freqs=[],[]
    fids=[]
    for fid in os.listdir(res_dir):
        fids.append(fid)
    fids.sort()
    for fid in fids:
        if 'json' in fid:
            with open(os.path.join(res_dir,fid)) as buf:
                res=json.load(buf)
            if 'wordswap' in res_dir:
                freqs.append(res['AVG_BIN'][1:])
            else:
                freqs.append(res['AVG_BIN'])
            lastbin.append(res['AVG_BIN'][-1])
        else:
            with open(os.path.join(res_dir,fid)) as buf:
                res=buf.readlines()[0].split(' ')
            freqs.append([float(s) for s in res[1:-1]])
            lastbin.append(float(res[-1]))

    return np.around(lastbin,3),np.array(freqs),fids
    
def get_spread(table):
    spread=(max(table[:,0])-min(table[:,0]))/(max(table[:,-1])-min(table[:,-1]))
    spread=round(spread,2)
    return spread

def get_all_avg(syn,infl,word,fids,tag):
    all_avg_0=np.around((syn[:,0]+infl[:,0])/2,2).reshape(-1,1)
    all_avg=np.around((syn[:,1:]+infl[:,1:]+word)/3,2)
    all_avg=np.concatenate((all_avg_0,all_avg),axis=1)
    print('Overall accuracy per model for increasing frequency bins')
    for i in range(len(fids)):
        print(fids[i],' '.join([str(e) for e in all_avg[i]]))
    avg_per_freqs=np.mean(all_avg,axis=0)
    #high_minus_low=avg_per_freqs[-1]-avg_per_freqs[0]
    #print('high-minus-low',tag,np.around(high_minus_low,3))
    return all_avg

def sort_along_bis(first,second):
    indices=np.argsort(first)
    first=first[indices]
    second=second[indices]

    pr=stats.pearsonr(first,second)
    sr=stats.spearmanr(first,second)
    print('pearson:',np.around([pr.statistic,pr.pvalue],2))
    print('spearman:',np.around([sr.statistic,sr.pvalue],2))

def sort_along_freq(source,tag):
    indices=np.arange(len(source[0]))
    if len(indices)==11:
        offset=1
    else:
        offset=0

    corr_bin=[]
    high_bin=source[:,-1]
    corr_bin=[]
    low_bin=np.mean(source[:,:1+offset],axis=1)
    res=stats.spearmanr(high_bin,low_bin)
    srl,svl=res.statistic,res.pvalue
    
    mid_bin=source[:,-2]
    res=stats.spearmanr(high_bin,mid_bin)
    srh,svh=res.statistic,res.pvalue
    corr_bin='/'.join([str(round(srl,2)),str(round(svl,2)),str(round(srh,2)),str(round(svh,2))])
    
    corr=[]
    for model in source:        
        res=stats.spearmanr(model,indices)
        sr,sv=res.statistic,res.pvalue
        corr.append(sr)
        
    #averaging correlation coefficient using inverse z score
    zs = [fisher_z(r) for r in corr]
    z_avg = np.mean(zs)
    r_avg = inverse_fisher_z(z_avg)
    return round(r_avg,2),corr_bin

def fisher_z(r):
        # Step 1: Convert to Fisher Z
    return 0.5 * np.log((1 + r) / (0.0000001+1 - r))

def inverse_fisher_z(z):
    return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)    

if __name__=='__main__': 
    #result directory from eval_longtail
    result_dir=sys.argv[1]
    wordswap_dir=os.path.join(result_dir,'wordswap')
    syntax_dir=os.path.join(result_dir,'agrswap')
    inflswap_dir=os.path.join(result_dir,'inflswap')
    
    syn,syn_per_freq,fids=get_score(syntax_dir,'agrswap')
    infl,infl_per_freq,_=get_score(inflswap_dir,'inflswap')
    word,word_per_freq,_=get_score(wordswap_dir,'wordswap')
    all_per_freq=get_all_avg(syn_per_freq,infl_per_freq,word_per_freq,fids,'ALL')

    syn_s=get_spread(syn_per_freq)
    infl_s=get_spread(infl_per_freq)
    word_s=get_spread(word_per_freq)
    all_s=get_spread(all_per_freq)
    print('')
    print('Spread ratios')
    print('overall:',all_s,'wordswap:',word_s,'inflswap:',infl_s,'agrswap:',syn_s)
    print('')
    syn_r,syn_r_bin=sort_along_freq(syn_per_freq,'agrswap')
    infl_r,infl_r_bin=sort_along_freq(infl_per_freq,'inflswap')
    word_r,word_r_bin=sort_along_freq(word_per_freq,'wordswap')
    all_r,all_r_bin=sort_along_freq(all_per_freq,'ALL')
    print('Spearman R between frequency bins and model perf, averaged across models')
    print('overall:',all_r,'wordswap:',word_r,'inflswap:',infl_r,'agrswap:',syn_r)
    