
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
    parser.add_argument("--wordlist",type=str,help='path to words list file for WordSwap',default='babylm-lt-swap/tmp_files_10M/longtail_wordlist')
    parser.add_argument("--output_file",type=str,help='path to sentence generation prompts file',default='babylm-lt-swap/tmp_files_10M/wordswap_sentence_prompts')
    return parser.parse_args(argv)

if __name__=='__main__':
    args=parse_arguments(sys.argv[1:])
    wordlist_file=args.wordlist
    output_file=args.output_file
    out=[]
    with open(wordlist_file) as buf:
        lines=buf.readlines()
    for line in tqdm.tqdm(lines):
        bin,word,pos,index,sentence=line.rstrip().split('|')
        base_pos=pos.split('_')[0].lower() 
        assert base_pos in ['noun','verb'],base_pos
        prompt=' '.join(("Given the",base_pos,"\'",word,"\'. Can you write a simple sentence that contains the",base_pos,"\'",word,"\' using at least 20 words. Make it simple. Write only this sentence between brackets."))
        out.append('|'.join((bin,word,pos,index,sentence,prompt)))
    
    with open(output_file,'w') as buf:
        buf.write('\n'.join(out)+'\n')