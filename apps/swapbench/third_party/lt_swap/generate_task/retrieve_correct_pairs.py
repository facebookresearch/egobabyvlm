
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os,sys
import tqdm
import numpy as np
import argparse

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",type=str,help='path to file with filtering answers',default='babylm-lt-swap/tmp_files_10M/wordswap_sentence_pairs_to_be_filtered')
    parser.add_argument("--output_file",type=str,help='path to output file with filtered sentence pairs',default='babylm-lt-swap/tmp_files_10M/wordswap_sentence_pairs_filtered')
    parser.add_argument("--task_type",type=str,help='wordswap,inflswap,agrswap',required=True)
    return parser.parse_args(argv)

def format_answer(g):
    print('->',g)
    start=g.rfind('[')
    end=g.rfind(']')
    if start==-1 or end==-1:
        if g in ['A','B']:
            print(g)
            print('')
            return g
        else:
            return None,None,None,None
    g=g[start+1:end]
    g=g.replace(' ','')
    print(g)
    print('')
    return g.upper()

def format_answers(ag1,ag11,ag2,ag22):
    
    ag1=format_answer(ag1)
    ag11=format_answer(ag11)
    ag2=format_answer(ag2)
    ag22=format_answer(ag22)
    print('_____')
    return ag1,ag11,ag2,ag22

# Run the async code
if __name__ == "__main__":
    args=parse_arguments(sys.argv[1:])
    input_file=args.input_file
    output_file=args.output_file
    task_type=args.task_type
    assert task_type in ['wordswap','inflswap','agrswap']
    with open(input_file) as buf:
        lines=buf.readlines()
    correct_pairs=[]
    pairs={}
    accepted,accepted_pos,all_pairs,all_pairs_pos={},{},{},{}
    h=[]
    for i in tqdm.tqdm(range(len(lines))):
        bin,original_pos,w1,s1,i1,g1,ig1,w2,s2,i2,g2,ig2,p1,p11,p2,p22,a1,a11,a2,a22,ag1,ag11,ag2,ag22=lines[i].rstrip().split('|')
        assert w1!=w2
        h.append(int(bin))
        pos=original_pos.split('_')[0]
        print(w1,w2,g1,g2)
        ag1,ag11,ag2,ag22=format_answers(ag1,ag11,ag2,ag22)
        if bin not in accepted:
            accepted[bin]=0
            all_pairs[bin]=0
            accepted_pos[bin]={}
            all_pairs_pos[bin]={}
        if pos not in accepted_pos[bin]:
            accepted_pos[bin][pos]=0
            all_pairs_pos[bin][pos]=0
        #continue
        if a1==ag1 and a11==ag11:
            if a2==ag2 and a22==ag22:
                key=[w1,w2]
                key.sort()
                key='-'.join(key)
                if key in pairs:
                    #print(i,w1,w2) #pair of word present twice
                    pass
                else:
                    pairs[key]=(g1,g2)
                correct_pairs.append('|'.join((bin,original_pos,w1,g1,ig1,w2,g2,ig2)))
                #correct_pairs.append('|'.join((bin,original_pos,w1,s1,i1,g1,ig1,w2,s2,i2,g2,ig2)))
                accepted[bin]+=1
                accepted_pos[bin][pos]+=1
        all_pairs[bin]+=1
        all_pairs_pos[bin][pos]+=1
    for bin in accepted:
        print(bin,':',round(100*float(accepted[bin]/all_pairs[bin]),2),'% accepted',accepted[bin])

    print(output_file,len(correct_pairs),'out of',len(lines))
    with open(os.path.join(output_file),'w') as buf:
        buf.write('\n'.join(correct_pairs)+'\n')