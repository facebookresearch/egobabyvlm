
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import tqdm
import random
import os
import torch
from utils import model_init, get_probs, pretty_print_avg
os.environ["TOKENIZERS_PARALLELISM"]='false'
import sys,json, argparse

def inference(inputs,model,tokenizer,loss_fn,verbose=True,bert=None):
    div=4
    batch_size=100*div
    assert len(inputs)%div==0
    assert len(inputs)==len(bins)*div
    assert batch_size%div==0
    
    #dictionnaries that stores results for pretty_print
    freq_bins=np.array([0,1,2,4,8,16,32,64,128,256,512,np.inf])
    for bin in range(len(freq_bins)-1):
        success[bin]=0
        all_pairs[bin]=0
        pos_success[bin]={}
        pos_all_pairs[bin]={}
        for pos_tag in ['VERB','NOUN','LONG','SHORT']:
            pos_success[bin][pos_tag]=0
            pos_all_pairs[bin][pos_tag]=0

    for i in tqdm.tqdm(range(int(len(inputs)/batch_size)+1)):
        batch=inputs[i*batch_size:(i+1)*batch_size]
        if len(batch)==0:
            continue
        batch_context=contexts[i*batch_size:(i+1)*batch_size]
        bin_batch=bins[i*int(batch_size/div):(i+1)*int(batch_size/div)]
        batch_log_probs=get_probs(model,tokenizer,batch,loss_fn,batch_context,cuda,bert)
        batch_log_probs=batch_log_probs.reshape(-1,div)
        
        for j in range(len(batch_log_probs)):
            prob_g1,prob_g2,prob_b1,prob_b2=batch_log_probs[j]
            bin,pos=bin_batch[j][:2]
            if prob_g1>prob_b1:
                success[bin]+=1
                pos_success[bin][pos]+=1
            if prob_g2>prob_b2:
                success[bin]+=1
                pos_success[bin][pos]+=1
            all_pairs[bin]+=2
            pos_all_pairs[bin][pos]+=2
        if verbose and (i-5)%15==0:
            pretty_print_avg(pos_success,pos_all_pairs,verbose) 
    cout=pretty_print_avg(pos_success,pos_all_pairs,verbose) 

    return cout
    
def swap_words(w1,ig1,g1,w2,ig2,g2):
    try:
        assert g1[ig1:ig1+len(w1)].lower()==w1,(w1,g1[ig1:ig1+len(w1)])
        assert g2[ig2:ig2+len(w2)].lower()==w2,(w2,g2[ig2:ig2+len(w2)])
        gg1=g1[:ig1]+w2+g1[ig1+len(w1):]
        gg2=g2[:ig2]+w1+g2[ig2+len(w2):]
    except:
        gg1,gg2=g1.split(' '),g2.split(' ')
        assert gg1[ig1].lower()==w1,(w1,g1)
        assert gg2[ig2].lower()==w2,(w2,g2)
        gg1[ig1],gg2[ig2]=w2,w1
        gg1,gg2=' '.join(gg1),' '.join(gg2) 
    
    #if the swapped word is the first one, the sentence may not start with upper case anymore
    gg1=gg1[0].upper()+gg1[1:]
    gg2=gg2[0].upper()+gg2[1:]
    return gg1,gg2


def parse_arguments(argv):
    #add your model in this list and edit model_init() in utils.py 

    task_types=['wordswap','inflswap','agrswap']
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type",type=str,choices=task_types,default='wordswap')
    parser.add_argument("--pair_file",type=str,help='path to sentence pairs',default='babylm-lt-swap/wordswap_pairs_10M')
    parser.add_argument("--model_name",type=str,help='huggingface model name, edit model_init function in utils.py',default='babylm/opt-125m-strict-small-2023')
    parser.add_argument("--output_dir",type=str,help='path to store results',default='babylm-lt-swap/results/')
    parser.add_argument("--verbose",help='output scores in console',action='store_true')
    parser.add_argument("--use_prefix",help='use prefix method, only for WordSwap',action='store_true')
    parser.add_argument("--tokenizer_file",help='tokenizer vocabulary',default=".")
    return parser.parse_args(argv)

if __name__ == '__main__':
    args=parse_arguments(sys.argv[1:])
    task_type=args.task_type
    pair_file=args.pair_file
    model_name=args.model_name
    output_dir=args.output_dir
    use_prefix=args.use_prefix
    verbose=args.verbose
    tokenizer_file=args.tokenizer_file
    if torch.cuda.is_available():
        cuda=True
    else:
        cuda=False
    output_dir=os.path.join(output_dir,task_type)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if verbose:
        print('output directory:',output_dir)
        if task_type=='wordswap':
            print('using prefix-method:',use_prefix)
    pairs={}
    with open(pair_file) as buf:
        pairs=buf.readlines()
    success,pos_success,all_pairs,pos_all_pairs={},{},{},{}
    inputs,contexts,bins=[],[],[]
    selected_pairs,tmp_pairs=[],[]

    for p in range(len(pairs)):    
        pair=pairs[p].rstrip() 
        if task_type=='wordswap':
            bin,rule,w1,s1,i1,g1,ig1,w2,s2,i2,g2,ig2=pair.split('|')
            assert rule in ['NOUN','NOUN_P','VERB','VERB_Past','VERB_PresT','VERB_PresC']
            rule=rule.split('_')[0]
        elif task_type=='inflswap':
            bin,rule,w1,g1,ig1,w2,g2,ig2=pair.split('|')
            assert rule in ['NOUN','VERB']
            assert not use_prefix
        elif task_type=='agrswap':
            bin,rule,w1,g1,ig1,w2,g2,ig2=pair.split('|')
            assert rule in ['ANAPHORALONG','ANAPHORASHORT','DET','SVLONG','SVSHORT']
            if 'SHORT' in rule or 'DET' in rule:
                rule='SHORT'
            elif 'LONG' in rule:
                rule='LONG'
            assert not use_prefix
        

        ig1,ig2=int(ig1),int(ig2)
        bin=int(bin)
        
        gg1,gg2=swap_words(w1,ig1,g1,w2,ig2,g2)    
       
        sentence_good_1,sentence_good_2=g1,g2
        sentence_bad_1,sentence_bad_2=gg1,gg2

        if use_prefix:
            context_good_1,context_good_2=s1,s2
            context_bad_1,context_bad_2=s2,s1
            sentence_good_1=' '.join((context_good_1,sentence_good_1))
            sentence_bad_1=' '.join((context_bad_1,sentence_bad_1))
            sentence_good_2=' '.join((context_good_2,sentence_good_2))
            sentence_bad_2=' '.join((context_bad_2,sentence_bad_2))
        else:
            context_good_1,context_good_2='',''
            context_bad_1,context_bad_2='',''

            #adding context=' ' or ' .' strangely helps opt-125 by quite a lot on BLIMP
            #sentence_good_1=context_good_1+sentence_good_1
            #sentence_bad_1=context_bad_1+sentence_bad_1
            #sentence_good_2=context_good_2+sentence_good_2
            #sentence_bad_2=context_bad_2+sentence_bad_2

        contexts+=[context_good_1,context_good_2,context_bad_1,context_bad_2]
        inputs+=[sentence_good_1,sentence_good_2,sentence_bad_1,sentence_bad_2]  
        bins.append((bin,rule,w1,w2,sentence_good_1,sentence_bad_1,sentence_good_2,sentence_bad_2))
        tmp_pairs.append(pair)
        
    
    model, tokenizer, loss_fn, model_type = model_init(model_name, cuda, tokenizer_file)  
    if verbose:
        print("Model init",model_name,"with vocab size:",tokenizer.vocab_size)
    #else:
    #    print(model_name)
        
    cout=inference(inputs,model,tokenizer,loss_fn,verbose,model_type)
    
    if verbose:
        print('Model:',model_name)
        print('pairs file:',pair_file)
        if task_type=='wordswap':
            print('using prefix-method:',use_prefix)

    base_model_name=model_name.split('/')[-1]
    cout['MODEL']=base_model_name
    with open(os.path.join(output_dir,base_model_name+".json"),'w') as buf:
        json.dump(cout,buf)
    print(task_type,' '.join([str(v) for v in cout['AVG_BIN']]))
