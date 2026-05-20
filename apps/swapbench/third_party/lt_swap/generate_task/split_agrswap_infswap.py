
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os,sys

output_last_filtering_step=sys.argv[1] #agreementswap and inflection swap are both in the output of the last
                                        #filtering step
inflection_swap_file=sys.argv[2] #output file that will contain inflectionswap
agreement_swap_file=sys.argv[3] #output file that will contain agreementswap

inflpairs,agrpairs=[],[]
with open(output_last_filtering_step) as buf:
    for line in buf:
        line=line.rstrip()
        rule=line.split('|')[1]
        if rule in ['VERB','NOUN']:
            inflpairs.append(line)
        elif rule in ['SVSHORT','SVLONG','ANAPHORASHORT','ANAPHORALONG','DET']:
            agrpairs.append(line)
        else:
            assert False,(rule,line)
        
with open(inflection_swap_file,'w') as buf:
    buf.write('\n'.join(inflpairs)+'\n')
with open(agreement_swap_file,'w') as buf:
    buf.write('\n'.join(agrpairs)+'\n')