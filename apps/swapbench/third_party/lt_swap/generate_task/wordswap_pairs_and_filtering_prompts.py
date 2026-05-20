
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os,sys,tqdm
import random, nltk
import numpy as np
import argparse

def make_prompt(s,gA,gB):
    prompt=("I have invented one new english word \'blick\' that you can use as in the following sentence:",
            "<start of sentence> "+s+"<end of sentence>",
            "Now I give you two new sentences A and B:",
            "<start of sentence A> "+gA+"<end of sentence A>",
            "<start of sentence B> "+gB+"<end of sentence B>",
            "Which of the sentence A or B uses the word 'blick' correctly? Put your answer, A or B, in between brackets.")
    return '_'.join(prompt)

def find_index_and_lower_case(sentence,word,vocabulary):
    #find index of target word and lower case it in the generated sentence
    index=None
    sentence=nltk.word_tokenize(sentence)
    for i in range(len(sentence)):
        w=sentence[i].lower()
        if len(w)>2 and vocabulary is not None and w not in vocabulary:
                return None,None
        if w==word:
            #lower casing target word
            sentence[i]=sentence[i].lower()
            if index is not None:
                #word is present twice in the context
                return None,None
            index=i   
    return index,' '.join(sentence)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",type=str,help='path to file with generated sentences',default='babylm-lt-swap/tmp_files_10M/wordswap_sentence_generations')
    parser.add_argument("--output_file",type=str,help='path to output file with llm outputs',default='babylm-lt-swap/tmp_files_10M/wordswap_sentence_pairs_with_filtering_prompts')
    parser.add_argument("--voc_file",type=str,help='path to the vocabulary file',default='babylm-lt-swap/tmp_files_10M/vocabulary')
    return parser.parse_args(argv)


# Run the async code
if __name__ == "__main__":
    args=parse_arguments(sys.argv[1:])
    input_file =args.input_file
    output_file=args.output_file
    voc_file=args.voc_file
    use_vocabulary_file=True #set this to True to filter out generations that
                             #contains words not present in the pretraining set
    max_pairs_per_pos=2000
    
    if use_vocabulary_file:
        vocabulary={}
        with open(voc_file) as buf:
            lines=buf.readlines()
            for line in lines:
                word,freq=line.rstrip().split(' ')
                assert word not in vocabulary
                vocabulary[word]=int(freq)
    else:
        vocabulary=None
    c=0
    are_inflections,words,inflections=set(),{},{}
    seen_words=set() #for some reason some base words are duplicated
    with open(input_file) as buf:
        lines=buf.readlines()
    for line in tqdm.tqdm(lines):
        line=line.rstrip().split('|')
        bin,word,pos,index_sentence,sentence,generation=line
        #formatting the generated sentence from the llm
        start=generation.rfind('[')
        end=generation.rfind(']')
        if start==-1 or end==-1:
            continue
        generation=generation[start+1:end]
        
        #getting the index of word in generation and put it in lower case in case it is not
        index_generation,generation=find_index_and_lower_case(generation,word,vocabulary)
        if index_generation is None:
            
            #there is an unknown word in the generation, or target word is present twice
            continue
        if bin not in words:
            words[bin]={}
        if pos not in words[bin]:
            words[bin][pos]=[]
        words[bin][pos].append((word,int(index_sentence),sentence,index_generation,generation))
    
    wordswap_list=[]
    wordpairs=set()
    for bin in words:
        for pos in words[bin]:
            tmp=[]
            for i in range(len(words[bin][pos])-1): 
                w1,i1,s1,ig1,g1=words[bin][pos][i]
                for j in range(i+1,len(words[bin][pos])):
                    
                    w2,i2,s2,ig2,g2=words[bin][pos][j]
                    if w1==w2:
                        continue
                    #if new pair add it to the output
                    key=[w1,w2]
                    key.sort()
                    key='-'.join(key)
                    if key in wordpairs:
                        continue
                    wordpairs.add(key)
                    #adding the prompts for the last filtering step

                    #creating new sentence with a new word inside
                    ss1,ss2,gg1,gg2=s1.split(' '),s2.split(' '),g1.split(' '),g2.split(' ')
                    #print(i1,ig1)
                    assert ss1[i1].lower()==w1 and gg1[ig1]==w1,(ss1,gg1,i1,ig1,w1)
                    assert ss2[i2].lower()==w2 and gg2[ig2]==w2,(ss2,gg2,i2,ig2,w2)

                    #changing w1 and w2 into blick
                    ss1[i1],gg1[ig1],ss2[i2],gg2[ig2]='blick','blick','blick','blick'
                    ss1,gg1,ss2,gg2=' '.join(ss1),' '.join(gg1),' '.join(ss2),' '.join(gg2)

                    #checking that the word is not in the modified sentences
                    #in some cases the target wod is present twice in the original or generated sentences
                    if w1 in ss1 or w1 in gg1:
                        continue
                    if w2 in ss2 or w2 in gg2:
                        continue

                    prompt1=make_prompt(ss1,gg1,gg2)
                    answer1='A'
                    prompt11=make_prompt(ss1,gg2,gg1)
                    answer11='B'

                    prompt2=make_prompt(ss2,gg2,gg1)
                    answer2='A'
                    prompt22=make_prompt(ss2,gg1,gg2)
                    answer22='B'

                    prompts_list='/'.join((prompt1,prompt11,prompt2,prompt22,answer1,answer11,answer2,answer22))
                    tmp.append('|'.join((str(bin),pos,w1,s1,str(i1),g1,str(ig1),w2,s2,str(i2),g2,str(ig2),prompts_list)))
            random.shuffle(tmp)
            wordswap_list+=tmp[:max_pairs_per_pos]
            print(bin,pos,len(tmp[:max_pairs_per_pos]))
    print('number of sentence pairs:',len(wordswap_list))
    with open(output_file,'w') as buf:
        buf.write('\n'.join(wordswap_list)+'\n')
