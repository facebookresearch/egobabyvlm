
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os,sys
from preprocessing_utils import format_word, space_characters, format_pos
from spellchecker import SpellChecker
import multiprocessing
import nltk
import json
import argparse

def format_context_sentence(sentence,accepted_chars):
    new_sentence=[]
    #checking that context sentences don t have web metadata and
    #removing not allowed symbols or very long words 
    for word in sentence:
        if 'javascript' in word or 'html' in word:
            return None
        if 'http' in word or 'www' in word:
            return None
        if len(word)>40:
            return None
        for char in word:
            if not char.isalpha() and char not in accepted_chars and char not in ["``","''"]:
                #removing non letter and not an allowed symbol
                char=' '
            new_sentence.append(char)
        new_sentence.append(' ')
    #removing unwanted white spaces
    new_sentence=''.join(new_sentence)
    new_sentence=' '.join(list(filter(None,new_sentence.split(' '))))
    new_sentence=new_sentence.strip()
    return new_sentence.split(' ')


def find_index(context,word):
    index=None
    for i in range(len(context)):
        w=context[i]
        if w.lower()==word:
            if index is not None:
                #word is present twice in the context
                return None
            index=i   
    return index

def trim_and_get_word_index(context,word,max_context_len):
    #find index of word in context
    index=find_index(context,word)
    #getting max allowed context on both side of the target word
    if index is not None and len(context)>max_context_len:
        start_ind=max(0,index-int(max_context_len/2))
        end_ind=min(len(context),index+int(max_context_len/2))
        missing_words=max_context_len-(end_ind-start_ind)
        if missing_words>0:
            if start_ind==0:
                end_ind+=missing_words
            else:
                start_ind-=missing_words
        context=context[start_ind:end_ind]
        assert len(context)>(max_context_len/2)
        assert len(context)<=max_context_len+1
        index-=start_ind
      
    return context,index


def update_dict(line,map_letters,spell,char_dict,vocabulary):
    max_context_len=128 #maximum size of stored contex sentence
    line=line.strip()
    line=line.replace('\t',' ')
    sentences=line.split('.')
    for raw_sentence in sentences:
        # skipping word with useles
        #separating words from symbols, except '-'
        sentence=nltk.word_tokenize(raw_sentence)
        if len(sentence)==0:
            continue
        pos_sentence=nltk.pos_tag(sentence)
        contex_sentence=format_context_sentence(sentence,map_letters['accepted_chars'])
        for i in range(len(pos_sentence)):
            word,pos=pos_sentence[i]
            lower_word=word.lower()
            if lower_word not in vocabulary:
                vocabulary[lower_word]=0
            vocabulary[lower_word]+=1

            if word.isdigit() or len(word)<2: #not considering numbers, symbols, small words
                continue
            if len(spell.known([word]))==0 or pos in ['NNP','NNPS']:
                #the word form must belong to the english dict
                #also removing here most of the named entities
                continue
            if i!=0 and word[0].isupper():
                #removing named entities that were missed by the POS tagger
                continue
            #POS are placed in one of the seven accepted category: 
            #NOUN,NOUN_P,VERB,VERB_Past,VERB_PresC,VERB_PresT,UNK
            pos=format_pos(word,pos)
            #now getting rid of upper case information
            word=word.lower()

            if word not in char_dict:
                char_dict[word]={'freq':0,'POS':{}}
            char_dict[word]['freq']+=1
            if pos not in char_dict[word]['POS']:
                char_dict[word]['POS'][pos]={}
                char_dict[word]['POS'][pos]['freq']=0
                char_dict[word]['POS'][pos]['word_index']=None
                char_dict[word]['POS'][pos]['context']=None
                char_dict[word]['POS'][pos]['context_len']=0
            char_dict[word]['POS'][pos]['freq']+=1
            if contex_sentence is None:
                    continue
            #storing a sentence from dataset that contains word of interest
            current_context_len=char_dict[word]['POS'][pos]['context_len']
            if current_context_len<min(len(contex_sentence),max_context_len):
                trimmed_context,word_index=trim_and_get_word_index(contex_sentence,word,max_context_len)    
                if word_index is None:
                    #word is not present exactly once in the context sentence
                    continue
                #putting target word to lower case in sentence 
                trimmed_context[word_index]=trimmed_context[word_index].lower()  
                
                char_dict[word]['POS'][pos]['context']=' '.join(trimmed_context)  
                char_dict[word]['POS'][pos]['word_index']=word_index
                char_dict[word]['POS'][pos]['context_len']=len(trimmed_context)  


def get_word_list(args):
    path,fid,output_wordslist_dir=args
    print(fid)
    c=0
    char_dict={}
    vocabulary={}
    spell = SpellChecker()
    map_letters={}
    map_letters['letters']='abcdefghijklmnopqrstuvwxyz'
    map_letters['accepted_chars']='abcdefghijklmnopqrstuvwxyz!"$%&\`\'\’()*,-.–0123456789:;?@[]'
    with open(path) as buf:
        for line in buf:
            #adding in dict all words that belong to the English dictionnary
            #if symbols are around the word, we may either skip the word
            #or separate this word from the symbols
            update_dict(line,map_letters,spell,char_dict,vocabulary)
    output_file=os.path.join(output_wordslist_dir,fid)
    output_voc_file=os.path.join(output_wordslist_dir,fid+'.voc')
    with open(output_file,'w') as buf:
        buf.write(json.dumps(char_dict))
    with open(output_voc_file,'w') as buf:
        buf.write(json.dumps(vocabulary))
    print('saving',output_file,'with vocabulary size:',len(vocabulary))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=str,help='path to pretraining dataset dir containing text files',default='babylm-lt-swap/tmp_files_10M/train_10M/')
    parser.add_argument("--output_wordlists_dir",type=str,help='path to words list directory to be created',default='babylm-lt-swap/tmp_files_10M/wordlists/')
    parser.add_argument("--ncpus",type=int,help='number of cpus for parallel computing, one per text file maximum',default=5)
    return parser.parse_args(argv)

if __name__=='__main__':
    #list all words, computing their frequency and their frequency per POS
    #each word is checked independantly and separated by white space from neighboring symbols
    #some words are rejected altogether if contain illegal characters
    args=parse_arguments(sys.argv[1:])
    data=args.data
    output_wordlists_dir=args.output_wordlists_dir
    ncpus=args.ncpus
    
    if not os.path.isdir(output_wordlists_dir):
        os.makedirs(output_wordlists_dir)

    #get word list and POS for each fid.
    arguments=[]
    for fid in os.listdir(data):
        path=os.path.join(data,fid)
        arguments.append((path,fid,output_wordlists_dir))

    if ncpus==1:
        for argument in arguments:
            get_word_list(argument)
    else:
        with multiprocessing.Pool(processes=ncpus) as pool:
            pool.map(get_word_list, arguments) 
   
