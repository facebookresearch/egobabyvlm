
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os,sys,tqdm
import random, nltk
import numpy as np
from preprocessing_utils import check_capital_and_punc,find_reflexive,find_verb
import argparse


def format_agreement_sentences(g1,g2,w1,w2,ig1,ig2,rule):
    if len(g1.split(' '))<3 or len(g2.split(' '))<3:
        return None,None
    split_g1=g1.split(' ')
    assert split_g1[ig1]==w1
    split_g2=g2.split(' ')
    assert split_g2[ig2]==w2
    v1,v2=None,None
    if 'LONG' in rule:
        if 'that can be' not in g1 or 'that can be' not in g2:
            return None,None
    if 'ANAPHORA' in rule:
        #make sure that there is no third person verb in between the noun and reflexive pronoun
        if (w1[-1]=='s' and w2[-1]!='s') or (w1[-2:]=='es' and w2[-1]=='s'):
            pass
        elif (w2[-1]=='s' and w1[-1]!='s') or (w2[-2:]=='es' and w1[-1]=='s'):
            pass
        else:
            print('not singular,plural',rule,w1,w2)
            return None,None
        gg1,gg2=find_verb(g1,g2,w1,w2)
        if gg1 is not None:
            return None,None
        split_g1,split_g2=find_reflexive(split_g1,split_g2,w1,w2)
        if split_g1 is None:
            return None,None
        
        if 'is' in split_g1 or 'are' in split_g1 or 'has' in split_g1 or 'have' in split_g1 or 'was' in split_g1 or 'were' in split_g1:    
            return None,None
        if 'is' in split_g2 or 'are' in split_g2 or 'has' in split_g2 or 'have' in split_g2 or 'was' in split_g2 or 'were' in split_g2:
            return None,None
        if split_g1[0].lower() in ['this','those','these'] or split_g2[0].lower() in ['this','those','these']:
            return None,None
        g1=' '.join(split_g1)
        g2=' '.join(split_g2)

    elif 'DET' in rule:
        if (w1[-1]=='s' and w2[-1]!='s') or (w1[-2:]=='es' and w2[-1]=='s'):
            pass
        elif (w2[-1]=='s' and w1[-1]!='s') or (w2[-2:]=='es' and w1[-1]=='s'):
            pass
        else:
            print('not singular,plural',rule,w1,w2)
            return None,None
        if ig2==0 or ig1==0:
            return None,None
        if split_g1[ig1-1].lower() not in ['that','these','this','those']:
            return None,None
        if split_g2[ig2-1].lower() not in ['that','these','this','those']:
            return None,None
        split_g1=split_g1[ig1-1:ig1+1]
        split_g2=split_g2[ig2-1:ig2+1]
        g1=' '.join(split_g1)
        g2=' '.join(split_g2)
        ig1=1
        ig2=1
        if 'itself' in split_g1 or 'herself' in split_g1 or 'himself' in split_g1 or 'themselves' in split_g1:
            return None,None
        if 'itself' in split_g2 or 'herself' in split_g2 or 'himself' in split_g2 or 'themselves' in split_g2:
            return None,None

    elif 'SV' in rule:
        #we ll have to make sure there are no reflexive pronouns 
        #and that there is a verb in third person / first person after the noun
        if (w1[-1]=='s' and w2[-1]!='s') or (w1[-2:]=='es' and w2[-1]=='s'):
            pass
        elif (w2[-1]=='s' and w1[-1]!='s') or (w2[-2:]=='es' and w1[-1]=='s'):
            pass
        else:
            print('not singular,plural',rule,w1,w2)
            return None,None
        if 'itself' in split_g1 or 'herself' in split_g1 or 'himself' in split_g1 or 'themselves' in split_g1:
            return None,None
        if 'itself' in split_g2 or 'herself' in split_g2 or 'himself' in split_g2 or 'themselves' in split_g2:
            return None,None
        if split_g1[0].lower() in ['this','those','these']:
            return None,None
        if split_g2[0].lower() in ['this','those','these']:
            return None,None
        g1,g2=find_verb(g1,g2,w1,w2)
        if g1 is None:
            return None,None

    
    
    #coarse removal of some common plural markers
    for pattern in ['multiple', 'several','many']:
        if pattern in g1:
            g1=g1.split(' ')
            ind=g1.index(pattern)
            g1=g1[:ind]+g1[ind+1:]   
            g1=' '.join(g1) 
            if ind<ig1:
                ig1=ig1-1
        if pattern in g2:
            g2=g2.split(' ')
            ind=g2.index(pattern)
            g2=g2[:ind]+g2[ind+1:]
            g2=' '.join(g2) 
            if ind<ig2:
                ig2=ig2-1
    return g1,g2

def make_prompt(s1,s2):
    prompt = (
    "Given the two sentences A and B:",
    "<start of sentence A> "+s1+" <end of sentence A>",
    "<start of sentence B> "+s2+" <end of sentence B>",
    "Which of the two sentences A or B is syntactically correct? Write your answer (A or B) in between brackets."
    )
    return ' '.join(prompt)


def find_index_and_target_word(sentence,w1,w2,vocabulary):
    #sentence contains either w1 or w2, this function finds the target word
    #put that word in lower case, and return None if unkown word is used
    index=None
    sentence=nltk.word_tokenize(sentence)
    for i in range(len(sentence)):
        w=sentence[i].lower()
        if len(w)>2 and vocabulary is not None and w not in vocabulary:
            return None,None,None
        if w in [w1,w2]:
            
            if index is not None:
                #w1 and w2 are both present or one of them is present twice
                return None,None,None
            index=i  
            word=w
    if index is None:
        return None,None,None    
    #lower casing target word
    sentence[index]=sentence[index].lower() 
    return index,word,' '.join(sentence)

def find_two_generations(w1_tmp,w2_tmp,g,vocabulary,rule):
    start=g.rfind('[')
    end=g.rfind(']')
    if start==-1 or end==-1:
        return None,None,None,None,None,None
    g=g[start+1:end]
    g=g.replace('\\','')
    g=g.replace('\"','')
    g=g.replace('\'','')   
    #removing empty space
    g=' '.join(list(filter(None, g.split(' '))))
    g1,g2=None,None
    #the pattern that enables to split the sentence is not always a period.
    for pattern in ['.','!','?','/',', but',', while',', whereas',', and ',',',';']:
        #final period is not a separator
        if pattern in g[:-1]:
            ind=g.find(pattern)
            g2=g[ind+len(pattern)+1:]
            g1=g[:ind]
            if len(g1)<3 or len(g2)<3:
                continue
            else:
                break
            
    if g1 is None or len(g1)==0 or g2 is None or len(g2)==0:
        return None,None,None,None,None,None
    ig1,w1,g1=find_index_and_target_word(g1,w1_tmp,w2_tmp,vocabulary)
    ig2,w2,g2=find_index_and_target_word(g2,w1_tmp,w2_tmp,vocabulary)
   
    if ig1 is None or ig2 is None or w1==w2:
        return None,None,None,None,None,None
  
    
    if rule not in ['VERB','NOUN']:
        
        g1,g2=format_agreement_sentences(g1,g2,w1,w2,ig1,ig2,rule)
        if g1 is None:
            return None,None,None,None,None,None
        #sentences have been changed a bit
        ig1,_,_=find_index_and_target_word(g1,w1,w1,vocabulary)
        ig2,_,_=find_index_and_target_word(g2,w2,w2,vocabulary)
        if ig1 is None or ig2 is None:
            return None,None,None,None,None,None
    #g1 and g2 must start and finish by capital letter and period
    #also checking that w1 and w2 are correctly placed.
    w1,w2,g1,g2=check_capital_and_punc(w1,w2,g1,g2,ig1,ig2)
    return w1,g1,ig1,w2,g2,ig2

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",type=str,help='path to sentence pair generations for syntactic tasks',default='babylm-lt-swap/tmp_files_10M/syntax_sentence_generations')
    parser.add_argument("--output_file",type=str,help='path to sentence pairs with filtering prompts filtered file',default='babylm-lt-swap/tmp_files_10M/syntax_sentence_generations_with_filtering_prompts')
    parser.add_argument("--voc_file",type=str,help='path to the vocabulary file',default='babylm-lt-swap/tmp_files_10M/vocabulary')
    return parser.parse_args(argv)

if __name__=='__main__':
    args=parse_arguments(sys.argv[1:])
    input_file=args.input_file
    output_file=args.output_file
    voc_file=args.voc_file
    use_vocabulary_file=True #set this to True to filter out generations that
                             #contains words not present in the pretraining set
    out=[]
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
    with open(input_file) as buf:
        lines=buf.readlines()
    seen_words=set() #for some reason some base words are duplicated
    sentence_pairs=set()
    for line in tqdm.tqdm(lines):
        try:
            bin,w1,p1,w2,p2,rule,generation=line.rstrip().split('|')
        except:
            continue
        tt1,tt2=w1,w2
        #formatting the generated sentence from the llm
        w1,g1,ig1,w2,g2,ig2=find_two_generations(w1,w2,generation,vocabulary,rule)
    
        if g1 is None:
            #finding the two sentences did not work
            continue
        key=[g1,g2]
        key.sort()
        key='-'.join(key)
        if key in sentence_pairs:
            continue
        sentence_pairs.add(key)
        #getting the index of word in generation
        #creating new sentence with a new word inside
        gg1,gg2=g1.split(' '),g2.split(' ')
        assert gg1[ig1]==w1,(gg1,ig1,w1,gg2,ig2,w2)
        assert gg2[ig2]==w2,(gg2,ig2,w2)

        #changing w1 and w2 into blick
        gg1[ig1],gg2[ig2]=w2,w1
        #checking that the word is not in the modified sentences
        #in some cases the target wod is present twice in the original or generated sentences
        assert w1 not in gg1,(w1,w2,gg1,gg2) 
        assert w2 not in gg2,(w1,w2,gg1,gg2) 
        gg1,gg2=' '.join(gg1),' '.join(gg2)
        #making prompt asking the LLM to solve the quadruplet
        prompt1=make_prompt(g1,gg1)
        answer1='A'
        prompt11=make_prompt(gg1,g1)
        answer11='B'

        prompt2=make_prompt(g2,gg2)
        answer2='A'
        prompt22=make_prompt(gg2,g2)
        answer22='B'
        prompts_list='/'.join((prompt1,prompt11,prompt2,prompt22,answer1,answer11,answer2,answer22))

        out.append('|'.join((str(bin),rule,w1,g1,str(ig1),w2,g2,str(ig2),prompts_list)))
        


    with open(output_file,'w') as buf:
        buf.write('\n'.join(out)+'\n')