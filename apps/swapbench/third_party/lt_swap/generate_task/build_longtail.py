
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os,sys
import ast
from preprocessing_utils import format_word
from spellchecker import SpellChecker
import tqdm
import nltk
import json
import argparse

def concat_voc(path,vocabulary):
    json.load
    with open(path) as buf:
        words=json.load(buf)
        for word in words:
            freq=words[word]
            if word not in vocabulary:
                vocabulary[word]=0
            vocabulary[word]+=int(freq)

def concat_dict(path,char_dict):
    json.load
    with open(path) as buf:
        words=json.load(buf)
        for word in words:
            h=words[word]
            if word not in char_dict:
                char_dict[word]={'freq':0,'POS':{}}
            char_dict[word]['freq']+=h['freq']
            for pos in h['POS']:
                if pos not in char_dict[word]['POS']:
                    char_dict[word]['POS'][pos]={}
                    char_dict[word]['POS'][pos]['freq']=0
                    char_dict[word]['POS'][pos]['context']=None
                    char_dict[word]['POS'][pos]['word_index']=None
                    char_dict[word]['POS'][pos]['context_len']=0
                char_dict[word]['POS'][pos]['freq']+=h['POS'][pos]['freq']
                h_context_len=h['POS'][pos]['context_len']
                current_context_len=char_dict[word]['POS'][pos]['context_len']
                if current_context_len<min(h_context_len,128):
                    char_dict[word]['POS'][pos]['context']=h['POS'][pos]['context']
                    char_dict[word]['POS'][pos]['context_len']=h_context_len
                    char_dict[word]['POS'][pos]['word_index']=h['POS'][pos]['word_index']
                
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--wordlists_dir",type=str,help='path to wordslist dir created by get_words_lists.py',default='babylm-lt-swap/tmp_files_10M/wordlists/')
    parser.add_argument("--output_wordlist",type=str,help='path to words list file for WordSwap',default='babylm-lt-swap/tmp_files_10M/longtail_wordlist')
    parser.add_argument('--output_inflpairs',type=str,help='path to words pairs file for InflSwap and AgrSwap',default='babylm-lt-swap/tmp_files_10M/longtail_inflpairs')
    parser.add_argument('--output_voc',type=str,help='path to vocabulary file',default='babylm-lt-swap/tmp_files_10M/vocabulary')
    return parser.parse_args(argv)



if __name__=='__main__':
    args=parse_arguments(sys.argv[1:])
    wordlists_dir=args.wordlists_dir
    output_wordlist_file=args.output_wordlist
    output_inflpairs_file=args.output_inflpairs
    output_voc_file=args.output_voc
    freq_bins=np.array([0,1,2,4,8,16,32,64,128,256,512,np.inf])
    char_dict={}
    words_per_bins={}
    longtail_morpho,longtail_infl=[],[]
    vocabulary={}
    for fid in os.listdir(wordlists_dir):
        path=os.path.join(wordlists_dir,fid)
        if 'voc'==fid.split('.')[-1]:
            concat_voc(path,vocabulary)
        else:
            concat_dict(path,char_dict)

    #saving concatenated vocabularies
    output=[]
    voc=dict(sorted(vocabulary.items(), key=lambda item: item[1]))
    for key in voc:
        output.append(key+' '+str(voc[key]))
    with open(output_voc_file,'w') as buf:
        buf.write('\n'.join(output)+'\n')

    print('saving vocabulary at',output_voc_file,len(output))
    print('nb keys',len(char_dict.keys()))
    spell = SpellChecker()
    voc={}
    skipped_words=0
    for form in tqdm.tqdm(char_dict):
        # A cluster is the set of all inflection+POS that comes from one base word form
        # Checking that among all POS tags and all inflections for that word, the frequency of that word 
        # with that POS tag is roughly the same as the total frequency of its cluster. 
        # intuitively, it means this word is particularly frequent in that inflection and POS tag.
        assert form not in voc
        most_common_pos_freq=0
        most_common_pos='UNK'
        #getting most frequent pos tag: noun,verb or UNK 
        for pos in char_dict[form]['POS']:
            pos_freq=char_dict[form]['POS'][pos]['freq']
            if pos_freq>most_common_pos_freq:
                most_common_pos_freq=pos_freq
                most_common_pos=pos
                context_for_most_common_pos=char_dict[form]['POS'][pos]['context']
                word_index_in_context=char_dict[form]['POS'][pos]['word_index']
        if most_common_pos=='UNK':
            #keeping only nouns and verbs
            continue
        #get inflections for this noun or verb
        inflections=format_word(form,most_common_pos,spell)
        if len(inflections)==0:
            #if word is too short it will not have any inflection
            continue

        cluster_freq=char_dict[form]['POS'][most_common_pos]['freq'] #sum of freq for all POS of that word
        cluster=[form,str(most_common_pos_freq)]
        most_common_pos_freq_bin=np.where(most_common_pos_freq>=freq_bins)[0][-1]
        for inflection,infl_pos in inflections:
            inflection_freq=0
            if inflection==form:
                continue
            if inflection in char_dict and infl_pos in char_dict[inflection]['POS']:
                inflection_freq=char_dict[inflection]['POS'][infl_pos]['freq']#adding the freq for all POS of that inflection
            
            cluster.extend([inflection,str(inflection_freq)])
            cluster_freq+=inflection_freq

            inflection_freq_bin=np.where(inflection_freq>=freq_bins)[0][-1]
            #taking the minimum bin of the two words
            inflection_freq_bin=min(inflection_freq_bin,most_common_pos_freq_bin)
            longtail_infl.append('|'.join((str(inflection_freq_bin),form,most_common_pos,inflection,infl_pos,form+' '+str(most_common_pos_freq)+' '+inflection+' '+str(inflection_freq))))
        if len(cluster)>2: #we have added at least one inflection to the cluster
            cluster_freq_bin=np.where(cluster_freq>=freq_bins)[0][-1]
            #checking that the freq for that word/POS is in the same bin as the sum
            #of frequencies of this word's inflections 
            if cluster_freq_bin!=most_common_pos_freq_bin:
                continue
        
        key=str(cluster_freq_bin)+'_'+most_common_pos.split('_')[0]
        if key not in words_per_bins:
            words_per_bins[key]=0
        if words_per_bins[key]>=2000:
            #if more than 2k words in that bin and POS, skipping
            continue
        words_per_bins[key]+=1 
        cluster=' '.join(cluster)
        if context_for_most_common_pos is None:
            #print('no context for :',form,most_common_pos,':',context_for_most_common_pos)
            #no found context or context did not pass filter from get_word_lists
            skipped_words+=1
            continue
        if len(context_for_most_common_pos.split(' '))<5:
            #print('too short context for :',form,most_common_pos,':',context_for_most_common_pos,char_dict[form])
            #word does not have enough context to be understand in its sentence
            skipped_words+=1
        
        
        longtail_morpho.append('|'.join((str(most_common_pos_freq_bin),form,most_common_pos,str(word_index_in_context),context_for_most_common_pos)))
    print('skipped words for wrong context:',skipped_words) 
    print('number of words for wordswap:',len(longtail_morpho))
    print('number of inflected pairs for agr/inflswap:',len(longtail_infl))
    with open(output_wordlist_file,'w') as buf:
        buf.write('\n'.join(longtail_morpho)+'\n')
    with open(output_inflpairs_file,'w') as buf:
        buf.write('\n'.join(longtail_infl)+'\n')
    print('saving at',output_wordlist_file,output_inflpairs_file)
