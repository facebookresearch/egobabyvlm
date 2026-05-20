
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
# Adding the parent directory to sys.path to avoid potential import errors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, PreTrainedTokenizerFast
import numpy as np
from generate_task.preprocessing_utils import format_pos
import ast, nltk, os
try:
    from unit_lm import UnitLM #slamkit usage
except:
    pass



def model_init(model_name, cuda,tokenizer_file):
  
    #if model_name is not a path, it has to belong to this list 
    allowed_models=['babylm/opt-125m-strict-2023','babylm/opt-125m-strict-small-2023',\
                'babylm/babyllama-100m-2024','babylm/babyllama-10m-2024',\
                'ltg/gpt-bert-babylm-base','ltg/gpt-bert-babylm-base',\
                'babylm/ltgbert-100m-2024','babylm/ltgbert-10m-2024',\
                'bg51717/antlm-bert-ntp_mlm-100m','bg51717/antlm-bert-ntp_mlm-10m',\
                'babylm/roberta-base-strict-2023','ltg/gpt-bert-babylm-base',\
                'SzegedAI/babylm-strict-mlsm','SzegedAI/babylm-strict-small-mlsm',\
                'SzegedAI/babylm24_LSM_strict','SzegedAI/babylm24_LSM_strict-small']

    if os.path.isdir(model_name):
        model_type='GPT'  
        #launching a model trained with Slamkit and a pretrained tokenizer 
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        model = UnitLM.from_pretrained(model_name,local_files_only=True,device_map = 'auto')
    else:
        assert model_name in allowed_models,(model_name,allowed_models)
        tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
        if model_name in ['babylm/opt-125m-strict-2023','babylm/opt-125m-strict-small-2023','babylm/babyllama-100m-2024','babylm/babyllama-10m-2024']:
            model_type='GPT'    
            model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True)              
        elif model_name in ['ltg/gpt-bert-babylm-base','ltg/gpt-bert-babylm-base']:
            model_type='GPT-BERT'
            model = AutoModelForMaskedLM.from_pretrained(model_name,trust_remote_code=True)
        elif model_name in ['babylm/ltgbert-100m-2024','babylm/ltgbert-10m-2024']:
            model_type='BERT'
            model = AutoModelForMaskedLM.from_pretrained(model_name,trust_remote_code=True)
        elif model_name in ['bg51717/antlm-bert-ntp_mlm-100m','bg51717/antlm-bert-ntp_mlm-10m']:
            model_type='BERT'
            model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True)
        elif model_name in ['babylm/roberta-base-strict-2023','babylm/roberta-base-strict-small-2023']: 
            model_type='BERT'
            model = AutoModelForMaskedLM.from_pretrained(model_name,trust_remote_code=True)
        elif model_name in ['SzegedAI/babylm-strict-mlsm','SzegedAI/babylm-strict-small-mlsm']:
            model_type='BERT'
            model = AutoModelForMaskedLM.from_pretrained(model_name,trust_remote_code=True)
        elif model_name in ['SzegedAI/babylm24_LSM_strict','SzegedAI/babylm24_LSM_strict-small']:          
            model_type='BERT'
            model = AutoModelForMaskedLM.from_pretrained(model_name,trust_remote_code=True)
        else:
            assert False,(model_name,': unknown model name')
        
    assert model_type in ['BERT','GPT','GPT-BERT']
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    model.eval()
    if cuda: 
        model.to('cuda')
        loss_fn=loss_fn.to('cuda')
    model.config.use_cache=False
    

    return model, tokenizer, loss_fn, model_type


def score_bert(model, tensor_input, attention_mask, mask_token_id,loss_fn, model_type, device,tokenizer,current_context_mask):
    
    nb_words=torch.sum(attention_mask).int()
    nb_context_words=torch.sum(current_context_mask).int()
    repeat_input = tensor_input.repeat(nb_words-1, 1)
    attention_mask = attention_mask.repeat(nb_words-1, 1)
    #masking the +1 diagonal
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:nb_words-1].to(device)
    masked_input = repeat_input.masked_fill(mask == 1, mask_token_id)

    if model_type=='BERT':
        #labels for regular BERT        
        labels = repeat_input.masked_fill( masked_input != mask_token_id, -100)
        
    elif model_type=='GPT-BERT':
        #labels for GPT-BERT
        repeat_labels = tensor_input.repeat(nb_words-1, 1)
        repeat_labels[:,:-1]=repeat_labels[:,1:]
        repeat_labels[:,-1]=tokenizer.encode(tokenizer.eos_token)[0]
        #getting the real diagonal
        label_mask = torch.ones(tensor_input.size(-1)).diag(0)[:nb_words-1].to(device)
        labels = repeat_labels.masked_fill( label_mask != 1, -100)
    else:
        assert False,model_type
    if nb_context_words>2:
        #assert False,(nb_context_words,current_context_mask)
        masked_input=masked_input[nb_context_words:,:]
        attention_mask=attention_mask[nb_context_words:,:]
        labels=labels[nb_context_words:,:]
    with torch.inference_mode():
        outputs = model(masked_input, attention_mask=attention_mask)

    logits=outputs['logits']
    batch_size,_,vocab_size=logits.size()  
    loss=loss_fn(logits.reshape(-1, vocab_size),labels.reshape(-1)).reshape(batch_size,-1)
    #there is only one sentence, we want one loss score
    loss=torch.sum(loss)#/torch.sum(mask)
    
    return loss

def get_loss(inputs,attention_masks,labels,loss_fn,model):
    with torch.no_grad():
        outputs = model(inputs,attention_mask=attention_masks)
    logits=outputs['logits']
    try:
        batch_size,_,vocab_size=logits.size()  
    except:
        batch_size,seq_len=inputs.size()
        logits=logits.reshape(batch_size,seq_len,-1)
        vocab_size=logits.size(-1)  
    loss=loss_fn(logits.reshape(-1, vocab_size),labels.reshape(-1)).reshape(batch_size,-1)
    return loss

def get_probs(model, tokenizer, sentences, loss_fn, contexts, cuda=False, model_type=None,norm_nll=False):
    if cuda:
        device='cuda'
    else:
        device='cpu'
    #tokenizer.pad_token = tokenizer.eos_token
    #if tokenizer.pad_token is None:
    #    tokenizer.pad_token ='[PAD]'
    #    tokenizer.eos_token ='[PAD]'
    #print(tokenizer.eos_token,tokenizer.pad_token)
    inputs=tokenizer(sentences, return_tensors='pt', padding=True)
    inputs['input_ids']=inputs['input_ids'].to(device)
    inputs['attention_mask']=inputs['attention_mask'].to(device)
    labels=inputs['input_ids'].clone()
    batch_size,seq_len=inputs['input_ids'].size()
    
    #padding context mask to the size of the sentence
    context_tokens=tokenizer(contexts, return_tensors='pt', padding=True,add_special_tokens=False)
    contexts_mask=torch.zeros((batch_size,seq_len)).to(device)
    mean_number_context_words=[]    
   
    for i in range(len(context_tokens['input_ids'])):
        #looping over sentences one by one
        assert torch.sum(inputs['attention_mask'][i])>torch.sum(context_tokens['attention_mask'][i])
        nb_context_tokens=torch.sum(context_tokens['attention_mask'][i]).int()+1 #adding one for the BOS/CLS token
        contexts_mask[i,:nb_context_tokens]=1
        mean_number_context_words.append(nb_context_tokens)
    mean_number_context_words=torch.tensor(mean_number_context_words)
    mean_number_context_words=torch.mean(mean_number_context_words.float()).int()
    
    if model_type!='GPT':
        loss=[]
        for i in range(len(sentences)):
            attention_mask=inputs['attention_mask'][i]
            current_context_mask=contexts_mask[i]
            assert torch.sum(attention_mask)>torch.sum(current_context_mask)
            tmp=score_bert(model,inputs['input_ids'][i],attention_mask,tokenizer.mask_token_id,loss_fn,model_type,device,tokenizer,current_context_mask,norm_nll)
            loss.append(tmp)
        log_probs=-torch.tensor([loss])
    else:
        labels=inputs['input_ids'].clone()
        labels[:,:-1]=inputs['input_ids'][:,1:]
        try:
            #if there is an eos token defined
            labels[:,-1]=tokenizer.encode(tokenizer.eos_token)[0]
        except:
            pass
        loss=get_loss(inputs['input_ids'],inputs['attention_mask'],labels,loss_fn,model)
        #intersection of non-padded BPEs and non-context BPEs
        #because we do not want to compute the loss on the context
        #nor on the padded tokens
        inputs_mask=inputs['attention_mask']
        if mean_number_context_words>2:
            inputs_mask=inputs_mask*(1-contexts_mask)
        #not computing loss on predicting end of sentence, it hads noise
        for i in range(len(loss)):
            index=len(tokenizer.encode(sentences[i]))
            inputs_mask[i,index-1]=0 #not predicting end of sentence to pad
            #could be useful for agreementswap not to predict the last token
            #inputs_mask[i,index-2]=0 #not predicting last word to end of sentence
        #applying mask and computing log probs 
        loss=loss*inputs_mask 
        denom=torch.sum(inputs_mask,dim=1)
        log_probs=-torch.sum(loss,dim=1)
        if norm_nll:
            log_probs/=denom
        #FOR DEBUG
        #for i in range(len(inputs['input_ids'])):
        #   print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][i]))
        
    assert not torch.isnan(torch.sum(log_probs)),(log_probs)
    assert not torch.isinf(torch.sum(log_probs)),(log_probs)
    return log_probs.cpu()
    

def read_pretraining_data(pretraining_file):
    with open(pretraining_file) as buf:
        lines=buf.readlines()
    freqs={}
    for line in tqdm.tqdm(lines):
        assert '|' not in line,line
        data=ast.literal_eval(line)
        freqs[data['word']]=data
        #adding inflections as entries without frequencies
        for inflection in data['all_inflections']:
            if inflection not in freqs:  
                pos=nltk.pos_tag([inflection])[0][1]
                pos=format_pos(inflection,pos)
                freqs[inflection]={'freq':0,'all_inflections':data['all_inflections'],'POS':{pos:{'freq':0}},'context':''}
               
    return freqs

def format_context(original_context,word,max_sentence_len=50):
    #get max number of words from sentence so that it surrounds 
    #target word
    context=original_context.lower().split(' ')
    assert word in context
    if len(context)>max_sentence_len:
        index=context.index(word)
        start_ind=max(0,index-int(max_sentence_len/2))
        end_ind=min(len(context),index+int(max_sentence_len/2))
        missing_words=max_sentence_len-(end_ind-start_ind)
        if missing_words>0:
            if start_ind==0:
                end_ind+=missing_words
            else:
                start_ind-=missing_words 
        original_context=original_context.split(' ')[start_ind:end_ind]
        assert len(original_context)>(max_sentence_len/2)
        assert len(original_context)<=max_sentence_len+1
        original_context=' '.join(original_context)
    return original_context

def get_context_util(word,pos,context_data,use_inflections=True):
    pos=format_pos(word,pos)
    if word not in context_data:
        #current word has been seen 0 times in the whole corpus
        #and is not an inflection of any known word in the corpus
        return None,None,0
    #if the word exists with this POS tag, lets use its frequency
    if pos in context_data[word]['POS']:
        freq=context_data[word]['POS'][pos]['freq']
    else:
        freq=0 
    
    #lets look if an inflection of the found word is MORE frequent
    if use_inflections:
        for inflection in context_data[word]['all_inflections']:
            if len(context_data[word]['all_inflections'])==1:
                continue
            base_pos=pos.split('_')[0] #VERBs can also be VERB_Past, VERB_PresT,...
            if pos in context_data[inflection]['POS']:
                infl_freq=context_data[inflection]['POS'][pos]['freq']
                if infl_freq>freq: #an inflection is (much) more common than the word
                    freq=infl_freq
                    word=inflection
            elif base_pos in context_data[inflection]['POS']:
                infl_freq=context_data[inflection]['POS'][base_pos]['freq']
                if infl_freq>freq: #an inflection is (much) more common than the word
                    freq=infl_freq
                    word=inflection
                    pos=base_pos
    return word,pos,freq

def get_context_util_nopos(word,pos,context_data,use_inflections=True):
    
    if word not in context_data:
        #current word has been seen 0 times in the whole corpus
        #and is not an inflection of any known word in the corpus
        return None,None,0
    freq=context_data[word]['freq']
    if use_inflections:
        #lets look if an inflection of the found word is MORE frequent
        for inflection in context_data[word]['all_inflections']:
            if len(context_data[word]['all_inflections'])==1:
                continue
            infl_freq=context_data[inflection]['freq']
            if infl_freq>freq: #an inflection is (much) more common than the word
                freq=infl_freq
                word=inflection

    pos=format_pos(word,pos)
    if pos not in context_data[word]['POS']:
        #find POS that is the most common
        most_common_pos=None
        most_common_pos_freq=0
        for pos in context_data[word]['POS']:
            pos_freq=context_data[word]['POS'][pos]['freq']
            if pos_freq>most_common_pos_freq:
                most_common_pos_freq=pos_freq
                most_common_pos=pos
        pos=most_common_pos
    return word,pos,freq

def get_context(context_data,sentence):
    
    pos_sentence=nltk.pos_tag(sentence.split(' '))
    min_freq=np.inf #words with high freq are considered as known
    min_freq_context=''
    min_freq_word=''
    min_freq_pos='UNK'
    #lets find the least frequent word in the sentence
    for word,pos in pos_sentence:
        word=word.lower()
        if len(word)<3:
            #one or two letter word are almost never informative
            continue
        word,pos,freq=get_context_util(word,pos,context_data)

        if freq==0:
            continue #no context exists for this one
         #along the sentence, lets keep only the least frequent word
        if freq>0 and freq<min_freq:
            min_freq=freq
            min_freq_word=word
            min_freq_pos=pos

    if min_freq_word!='':
        min_freq_context=context_data[min_freq_word]['POS'][min_freq_pos]['context']
        assert min_freq_word in min_freq_context.lower(),(min_freq_word,min_freq_context)
        #get a sentence centered on the target word
        min_freq_context=format_context(min_freq_context,min_freq_word,max_sentence_len=25)
        min_freq_context='( '+min_freq_context+' )'
    return min_freq_context,min_freq_word,min_freq


def pretty_print_avg(success,all_pairs,verbose):
   
    verbose_cout,matrix,variance={},{},{}
    for bin in success:
        for pos in success[bin]:
            if all_pairs[bin][pos]>0:
                if bin not in matrix:
                    matrix[bin]={}
                    variance[bin]={}
                    verbose_cout[bin]=[] 
                if success[bin][pos]>0:
                    tmp_res=float(success[bin][pos])/float(all_pairs[bin][pos])
                    tmp_std=np.sqrt(tmp_res*(1-tmp_res)/all_pairs[bin][pos])
                    matrix[bin][pos]=np.around(tmp_res,4)
                    variance[bin][pos]=np.around(tmp_std*tmp_std,4)
                if verbose:
                    verbose_cout[bin].append(' '.join(('bin:'+str(bin),pos+':',str(round(tmp_res,2))+'+/-'+str(round(tmp_std,2)),'nb pairs:',str(all_pairs[bin][pos]))))
    bins=[k for k in success.keys()]
    bins.sort() #need to add keys in order
    pos_tags=[k for k in matrix[list(matrix.keys())[0]].keys()]
    pos_tags.sort()
    
    sorted_matrix=[]
    for bin in bins:
        sorted_matrix.append([])
        for pos in pos_tags:
            if bin in matrix and pos in matrix[bin]:
                sorted_matrix[bin].append(matrix[bin][pos])
            else:
                sorted_matrix[bin].append(0)
    sorted_matrix=np.array(sorted_matrix)
    sorted_matrix[sorted_matrix == 0] = np.nan
    avg_per_bin=np.around(np.nanmean(sorted_matrix,axis=1),3)
    avg_per_pos=np.around(np.nanmean(sorted_matrix,axis=0),3)
   
   
    out={'AVG_BIN':list(avg_per_bin)}
    out['AVG_SUBTASK']=list(avg_per_pos)
    out['AVG']=np.around(np.nanmean(avg_per_bin),3)
    out['BINS']=list(bins)
    out['SUBTASK']=list(pos_tags)
    out['MATRIX']=matrix
    if verbose:
        cout=[]
        for bin in verbose_cout:
            cout.append(' '.join(verbose_cout[bin])) 
        cout.append('bin indices: '+' '.join([str(b) for b in bins])) 
        cout.append('subtasks: '+' '.join(pos_tags))       
        cout.append(' '.join(('Accuracy per bin:',str(avg_per_bin),'global average:',str(np.around(np.nanmean(avg_per_bin),3)))))
        cout.append(' '.join(('Accuracy per subtask:',str(avg_per_pos),'global average:',str(np.around(np.nanmean(avg_per_pos),3)))))
        print('\n'.join(cout))
        
    return out

def check_capital_and_punc(w1,w2,g1,g2,ig1,ig2):
    #word are lower case and sentences start with upper case
    w1=w1.lower()
    w2=w2.lower()
    g1=g1[0].upper()+g1[1:]
    g2=g2[0].upper()+g2[1:]
    #adding period at the end
    if g1[-1] not in ['.','!','?']:
        g1=g1+' .'
    if g2[-1] not in ['.','!','?']:
        g2=g2+' .'    
    if g1[0] in [',',' '] or g2[0] in [',',' ']:
        return None,None,None,None

    assert g1[-1] in ['.','!','?'],g1
    assert g2[-1] in ['.','!','?'],g2
    
    #there is no space in the sentences
    split_g1=g1.split(' ')
    split_g2=g2.split(' ')
    split_g1 = list(filter(None, split_g1))
    split_g2 = list(filter(None, split_g2))

    #the word are still in place
    assert ig1<len(split_g1),(split_g1,w1,ig1,split_g2,w2,ig2)
    assert ig2<len(split_g2),(split_g1,w1,ig1,split_g2,w2,ig2)
    assert split_g1[ig1].lower()==w1,(split_g1,w1,ig1,split_g2,w2,ig2)
    assert split_g2[ig2].lower()==w2,(split_g1,w1,ig1,split_g2,w2,ig2)
    g1=' '.join(split_g1)
    g2=' '.join(split_g2)
    return w1,w2,g1,g2