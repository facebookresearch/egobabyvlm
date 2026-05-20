
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import random
import tqdm
import os
from utils import get_probs, model_init
os.environ["TOKENIZERS_PARALLELISM"]='false'
import sys

def pretty_print(success,all_pairs,verbose=False):
    cout=[]
    all_successes=0
    all_pairs_value=0
    for bin in success:
        if all_pairs[bin]>0:
            res=round(success[bin]/all_pairs[bin],2)
            std=round(np.sqrt(res*(1-res)/all_pairs[bin]),3)
            all_successes+=success[bin]
            all_pairs_value+=all_pairs[bin]
            if verbose:
                cout.append(' '.join([str(bin),str(res),str(std),str(all_pairs[bin])]))
            else:
                cout.append(str(res))
    print('\n'.join(cout))
    res=all_successes/(0.00001+all_pairs_value)
    std=round(np.sqrt(res*(1-res)/(0.00001+all_pairs_value)),3)
    print('overall mean:',round(res,3),'+/-',str(std),all_pairs_value)
    cout.append(str(round(res,3)))
    return cout


if __name__ == '__main__':
    pairs_file=sys.argv[1]
    model_name=sys.argv[2]
    try:
        tokenizer_file=sys.argv[3]
    except:
        tokenizer_file=None
    norm_nll=True #normalize NLL by sentence length
    if torch.cuda.is_available():
        cuda=True
    else:
        cuda=False
    pairs={}
    freq_bins=np.array([0,1,2,4,8,16,32,64,128,256,512,np.inf])
    success,all_pairs={},{}
    for bin in range(len(freq_bins)):
        success[bin]=0
        all_pairs[bin]=0
    with open(pairs_file) as buf:
        pairs=buf.readlines()
    print('number of pairs:',len(pairs))
    inputs,contexts,bins=[],[],[]
    for p in tqdm.tqdm(range(len(pairs))):  
        pair=pairs[p].rstrip().split('|')
        if len(pair)==3:
            sentence_good,sentence_bad,bin=pair
            bin=0
        else:
            sentence_good,sentence_bad,bin,dataset=pair[:4]
        bin=int(bin) 
        
        context_good=''
        context_bad=''
        if context_good!='':
            #use context=' ' boosts performances on OPT especially on blimp
            #no idea why
            sentence_good=context_good+sentence_good
            sentence_bad=context_bad+sentence_bad
        contexts+=[context_good,context_bad]
        inputs+=[sentence_good,sentence_bad]   
        bins.append(bin)
    
    
    model, tokenizer, loss_fn, bert = model_init(model_name, cuda,tokenizer_file)  
    print("Model init",model_name,"with vocab size:",tokenizer.vocab_size)

    batch_size=200*2 #needs to be a multiple of 4 s
    assert len(inputs)%2==0
    assert len(inputs)==len(bins)*2
    assert batch_size%2==0
    nb_batches=int(len(inputs)/batch_size)+1
    print('batches:',nb_batches)
    for i in tqdm.tqdm(range(nb_batches)):
        batch=inputs[i*batch_size:(i+1)*batch_size]
        if len(batch)==0:
            break
        batch_context=contexts[i*batch_size:(i+1)*batch_size]
        bin_batch=bins[i*int(batch_size/2):(i+1)*int(batch_size/2)]
        batch_log_probs=get_probs(model,tokenizer,batch,loss_fn,batch_context,cuda,bert,norm_nll)
        batch_log_probs=batch_log_probs.reshape(-1,2)
        for j in range(len(batch_log_probs)):
            prob_g1,prob_g2=batch_log_probs[j]
            bin=bin_batch[j]
            if prob_g1>prob_g2:
                success[bin]+=1
            all_pairs[bin]+=1  
        if i%20==0:
            pretty_print(success,all_pairs,True)
    cout=pretty_print(success,all_pairs,True)        
    print('Model:',model_name)
    print('pairs file:',pairs_file)