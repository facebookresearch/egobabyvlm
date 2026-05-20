
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random, json, sys
import tqdm, argparse


def make_prompt_minimal(word,inflection,pos,rule):
    pos=pos.lower()+'s'
    adding=''
    if rule=='VERB':
        adding='that each uses one of these two verbs'
    elif rule=='NOUN':
        adding='that each uses one of these two nouns'
    elif rule=='DET':
        adding=' '.join(('that shows a determiner-noun agreement, using either that,these,this or those. For instance, using the nouns \'misconduct\' and \'misconducts\', you can write something like: \'This misconduct is a serious offense. These misconducts are serious offenses.\'. Now please do the same with \'',word,'\' and,\'',inflection,'\''))
    elif rule=='SVSHORT':
        adding='that show a short distance subject-verb agreement at the present tense. The subject and the verb must be placed close to each other'
    elif rule=='SVLONG':
        if pos=='VERB':
            adding=' '.join(('that shows a long distance subject-verb agreement through a relative clause starting by \'that can be\'. For instance, using the verbs \'let\' and \'lets\', you can write something like: \'The person that can be trusted lets the dog out. The people that can be trusted let the dog out.\'. Now please do the same with \'',word,'\' and,\'',inflection,'\''))
        else:
            adding=' '.join(('that shows a long distance subject-verb agreement through a relative clause starting by \'that can be\'. For instance, using the nouns \'neighbor\' and \'neighbors\', you can write something like: \'The neighbor that can be trusted lets his dog out. The neighbors that can be trusted let their dog out.\'. Now please do the same with \'',word,'\' and,\'',inflection,'\''))
    elif rule=='ANAPHORASHORT':
        adding=' '.join(('that shows a short distance usage of reflexive pronouns. The pronouns must be placed close to the subject \'',word,'\' and,\'',inflection,'\'. Please use the past tense'))
    elif rule=='ANAPHORALONG':
        adding=' '.join(('that shows a long distance usage of reflexive pronouns through a relative clause starting by \'that can be\'. For instance, using the verbs \'medecine\' and \'medecines\', you can write something like: \'The medecine that can be bought anywhere, proved itself to be very effective. The medecines that can be bought anywhere, proved themselves to be very effective.\'.  Now please do the same with \'',word,'\' and,\'',inflection,'\'')) 
    else:
        assert False,rule

    prompt=' '.join(("Using the",pos,"\'",word,"\' and,\'",inflection,"\', please write a minimal pair of sentences",adding,". You must encapsulate the two sentences together in between brackets."))
    return prompt

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inflpairs",type=str,help='path to inflected pairs filtering generation file',default='babylm-lt-swap/tmp_files_10M/inflpairs_to_be_filtered')
    parser.add_argument("--output_file",type=str,help='path to minimal sentence pairs prompts',default='babylm-lt-swap/tmp_files_10M/syntax_sentence_prompts')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args=parse_arguments(sys.argv[1:])
    input_file=args.inflpairs
    output_file=args.output_file
    with open(input_file) as buf:
        lines=buf.readlines()
    out=[]
    for line in tqdm.tqdm(lines):
        bin,word,pos,inflection,pos_infl,metadata,llm_output=line.rstrip().split('|')
        #formatting the generated sentence from the llm and checking it says 'yes'
        start=llm_output.rfind('[')
        end=llm_output.rfind(']')
        if start==-1 or end==-1 or end-start<=1:
            continue
        llm_output=llm_output[start+1:end]
        if llm_output.lower()!='yes':
            #the llm says this word pair cannot be used for syntactic tasks
            continue

        base_pos=pos.split('_')[0]
        if metadata=='areinflections':
            #word and inflection are indeed two inflections of the same word  
            prompt=make_prompt_minimal(word,inflection,base_pos,base_pos)
            out.append('|'.join((bin,word,pos,inflection,pos_infl,base_pos,prompt)))
        elif metadata=='issubject':
            #word and inflections can be the subject of a sentence
            assert base_pos=='NOUN',pos
            for rule in ['SVSHORT','SVLONG','ANAPHORASHORT','ANAPHORALONG','DET']:
                prompt=make_prompt_minimal(word,inflection,base_pos,rule)
                out.append('|'.join((bin,word,pos,inflection,pos_infl,rule,prompt)))
                out.append('|'.join((bin,word,pos,inflection,pos_infl,rule,prompt)))
                out.append('|'.join((bin,word,pos,inflection,pos_infl,rule,prompt)))
    
    print(output_file,len(out))
    with open(output_file,'w') as buf:
        buf.write('\n'.join(out)+'\n')
     
    