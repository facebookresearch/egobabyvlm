
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tqdm
import sys
import argparse

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inflpairs",type=str,help='path to inflected pairs file for AgrSwap and InflSwap',default='babylm-lt-swap/tmp_files_10M/longtail_inflpairs')
    parser.add_argument("--output_file",type=str,help='path to inflected pairs filtering prompts file',default='babylm-lt-swap/tmp_files_10M/inflpairs_filtering_prompts')
    return parser.parse_args(argv)

if __name__=='__main__':
    args=parse_arguments(sys.argv[1:])
    inflpairs_file=args.inflpairs
    output_file=args.output_file
    dictionnary={}
    out=[]
    
    with open(inflpairs_file) as buf:
        lines=buf.readlines()
    for line in tqdm.tqdm(lines):
        bin,word,pos,inflection,pos_infl,_=line.rstrip().split('|')
        base_pos=pos.split('_')[0].lower() 
        prompt=' '.join(("Given the two",base_pos+'s \''+word+'\' and \''+inflection+'\'',". Can you tell if they are two inflections of the same",base_pos,"? Answer by yes or no. Write your answer in between brackets."))
        out.append('|'.join((bin,word,pos,inflection,pos_infl,'areinflections',prompt)))
        if base_pos=='noun':
            #verbs are harder to control for AgreementSwap, we remove them from here
            if word[-1]=='s':
                singular=inflection
            else:
                singular=word
            prompt="Given the noun \'"+singular+"\'. Can this noun take a reflexive pronoun? Answer by yes or no. Write your answer in between brackets."
            out.append('|'.join((bin,word,pos,inflection,pos_infl,'issubject',prompt)))
    with open(output_file,'w') as buf:
        buf.write('\n'.join(out)+'\n')