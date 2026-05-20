
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os,sys
import re
import os,sys, tqdm
import numpy as np

def space_characters(word,map_letters):
    #space symbols from letters and compute the symbol ratio
    r=0
    new_word=''
    has_foreign_letters=False
    word=word.lower()
    for i in range(len(word)):
        char=word[i]

        if char in map_letters['foreigns']:
            has_foreign_letters=True
        
        if char not in map_letters['letters']: 
            if char not in '!"$%&\'()*,-.0123456789:;?@[]’':
                #it is a real symbol
                r+=1
                new_word+=char
            else:
                new_word+=' '+char+' '
        else:
            new_word+=char

    #remove beginning and trailing char
    new_word=' '.join(list(filter(None,new_word.split(' '))))
    new_word=new_word.strip()
    ratio_symbols=float(r)/float(len(word))
    return new_word,has_foreign_letters,ratio_symbols

def format_words(sentence,map_letters):
    #rejecting full sentences if does not pass quality filters

    new_sentence=[]
    for word in sentence:
        assert len(word)>0,sentence
        word,_,_=space_characters(word,map_letters)
        new_sentence.append(word)
    return ' '.join(new_sentence)

def format_line(chars,index=None,word=None):
    if index is None:
        sentence=list(filter(None,chars.split(' ')))
        sentence=format_words(sentence,map_letters)
        return sentence
    index=int(index)
    try:
        assert chars[index:index+len(word)].lower()==word,(word,chars,index)
    except:
        assert chars.split(' ')[index].lower()==word,(word,chars,index)
    chars=chars.replace('...','.')
    chars=chars.replace('\t',' ')
    sentence=list(filter(None,chars.split(' ')))
    sentence=format_words(sentence,map_letters)
    #print(sentence,word,index)
    index=sentence.split(' ').index(word)
    return sentence,str(index)

if __name__=='__main__':
    #1- removing sentences that contains headers, 
    # too many foreign words, URL, very long words, words with too many symbols
    #2- addings speaker tags in a consistant way across different dialogue datasets
    #3- numbers are split into individual figures
    #4- allowed symbols are spaced out from words: !"$%&\'()*,-.0123456789:;?@[] 
    #5- only '-' are kept attached to words for LMs to identify compound words

    
    data='babylm-lt-swap/'
    output_char_dir='babylm-lt-swap-lower-case/'
    if not os.path.isdir(output_char_dir):
        os.makedirs(output_char_dir)
    map_letters={}
    map_letters['letters']='abcdefghijklmnopqrstuvwxyz'
    map_letters['foreigns']='àáâãäåæçèéêëìíîïðñòóôõöøùúûüýÿāăąćĉčďđēĕėęěĝğġģĥħĩīĭįıĵķĺļľłńņňŋōŏőœŕŗřśŝşšţťũūŭůűŵŷźżžơưǎǐǒǔǚǧǩǫǵǹȁȃșțḋḍḏḗḡḥḩḫḱḵḷḻḿṁṃṅṇṉṓṗṙṛṟṣṫṭṯṳṹẏẓạảấầẩậắằẵặẹẻẽếềểễệỉịọỏốồổỗộớờởợụủứửữự'
    all_words=0
    for fid in os.listdir(data):
        path=os.path.join(data,fid)
        if not os.path.isfile(path):
            continue
        output_char_file=os.path.join(output_char_dir,fid)
        formatted_chars=[]
        print(fid)
        if os.path.isfile(output_char_file):
            continue
        c=0
        skipped={}
        fid=fid.split('.')[0]
        with open(path) as buf:
            lines=buf.readlines()
            for line in tqdm.tqdm(lines):
                if 'wordswap' in fid:
                    bin,rule,w1,s1,i1,g1,ig1,w2,s2,i2,g2,ig2=line.rstrip().split('|')
                    g1,ig1=format_line(g1,ig1,w1)
                    g2,ig2=format_line(g2,ig2,w2)
                    sentence='|'.join((bin,rule,w1,s1,i1,g1,ig1,w2,s2,i2,g2,ig2))
                elif 'blimp' in fid:
                    g1,g2,bin,rule=line.rstrip().split('|')[:4]
                    g1=format_line(g1)
                    g2=format_line(g2)
                    sentence='|'.join((g1,g2,bin,rule))
                else:
                    bin,rule,w1,g1,ig1,w2,g2,ig2=line.rstrip().split('|')
                    g1,ig1=format_line(g1,ig1,w1)
                    g2,ig2=format_line(g2,ig2,w2)
                    sentence='|'.join((bin,rule,w1,g1,ig1,w2,g2,ig2))
                formatted_chars.append(sentence)
              
        with open(output_char_file,'w') as buf:
            buf.write('\n'.join(formatted_chars)+'\n')
        print('total word accepted so far:',all_words)
        print('number sentences removed:',skipped)
    print('Final all kept words:',all_words)
