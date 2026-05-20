
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os,sys
from spellchecker import SpellChecker

def find_reflexive(split_g1,split_g2,w1,w2):
    pronouns=['himself','itself','herself','themselves']
    r1,r2=-1,-1
    for pronoun in pronouns:
        if pronoun in split_g1:
            r1=split_g1.index(pronoun)
        if pronoun in split_g2:
            r2=split_g2.index(pronoun)
    if r1==-1 or r2==-1:
        return None,None
    split_g1=split_g1[:r1+1]
    split_g2=split_g2[:r2+1]
    if w1 not in split_g1 or w2 not in split_g2:
        return None,None
    return split_g1,split_g2

def find_verb(g1,g2,w1,w2):
    #check if the verbs are variation of the verb 'be'
    #for subject-verb agreement we want the main verb to be 'be'
    #but for anaphora agreement we want the opposite as the verb is a singular/plural marker
    
    split_g1=g1.split(' ')
    split_g2=g2.split(' ')
    if (w1[-1]=='s' and w2[-1]!='s') or (w1[-2:]=='es' and w2[-1]=='s'):
        singular=split_g2
        plural=split_g1
    elif (w2[-1]=='s' and w1[-1]!='s') or (w2[-2:]=='es' and w1[-1]=='s'):
        singular=split_g1
        plural=split_g2
    else:
        assert False,(w1,w2)

    #finding i_s and i_p that are the index of the verbs
    i_s=-1
    vs,vp=None,None
    bes_dict={'are':'is','have':'has','do':'does','dont':'doesnt','arent':'isnt'}
    neg_dict={'are':'isnt','have':'hasnt','do':'doesnt','dont':'does','arent':'is'}
    for i_p in range(len(plural)):
        w=plural[i_p]
        if w in [w1,w2]:
            continue
        elif w in bes_dict:
            vp=w
            vs=bes_dict[w]
            if vs in singular:
                i_s=singular.index(vs)
            elif neg_dict[w] in singular:
                vs=neg_dict[w]
                i_s=singular.index(vs)
            else:
                #print(singular,plural,vs,vp)
                return None,None
            break
        elif w+'s' in singular:
            vp=w
            vs=w+'s'
            i_s=singular.index(vs)
            break
        elif w+'es' in singular:
            vp=w
            vs=w+'es'
            i_s=singular.index(vs)
            break
        elif w[-1]=='y' and w[:-1]+'ies' in singular:
            vp=w
            vs=w[:-1]+'ies'
            i_s=singular.index(vs)
            break

    if i_s==-1:
        #print(singular,plural,i_s,i_p)
        return None,None
    singular=singular[:i_s+1]
    plural=plural[:i_p+1]
    gs=' '.join(singular)    
    gp=' '.join(plural)
    if w1 in singular and w2 in plural:
        g1,g2=gs,gp
        v1,v2=vs,vp
    elif w1 in plural and w2 in singular:
        g1,g2=gp,gs
        v1,v2=vp,vs
    else:
        return None,None
    return g1,g2

def check_capital_and_punc(w1,w2,g1,g2,ig1,ig2,use_split=True):
    #word are lower case and sentences do not start with upper case
    #in case target word is in first position
    w1=w1.lower()
    w2=w2.lower()
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
    
    if use_split:
        #the word are still in place
        assert ig1<len(split_g1),(split_g1,w1,ig1,split_g2,w2,ig2)
        assert ig2<len(split_g2),(split_g1,w1,ig1,split_g2,w2,ig2)
        assert split_g1[ig1].lower()==w1,(split_g1,w1,ig1,split_g2,w2,ig2)
        assert split_g2[ig2].lower()==w2,(split_g1,w1,ig1,split_g2,w2,ig2)
    g1=' '.join(split_g1)
    g2=' '.join(split_g2)
    
    return w1,w2,g1,g2

def nn_inflections(noun):
  # Third-person singular present: add "s" or "es"
    if noun[-1] in ['s', 'x', 'z'] or noun[-2:] in ['sh', 'ch']:
        plural = noun + 'es'  # For nouns ending in 's', 'sh', 'ch', 'x', or 'z' (e.g., "pass" → "passes")
    elif noun[-1]=='y':
        plural = noun[:-1] + 'ies'
    else:
        plural = noun + 's'  # Regular case (e.g., "run" → "runs")

    return [(noun,'NOUN'),(plural,'NOUN_P')]

def vb_inflections(verb,spell):
    # Past tense: regular verbs add "ed"
    if verb[-1]=='e':
        past_tense = verb + 'd'  # For verbs ending in 'e' (e.g., "love" → "loved")
        present_participle = verb[:-1] + 'ing'  # Remove 'e' for verbs ending in 'e' (e.g., "love" → "loving")
    elif verb[-1]=='y':
        past_tense = verb[:-1] + 'ied' #study studied
        if len(spell.known([past_tense]))==0: 
            past_tense=verb + 'ed'#play played
        present_participle = verb + 'ing' #study studying
    elif len(verb) == 3 and verb[-1] not in 'aeiou':
        #double letter for three letter adj
        past_tense = verb + verb[-1] + 'ed'
        present_participle = verb + verb[-1] + 'ing'
    else:
        past_tense = verb + 'ed'  # Regular case for past tense (e.g., "talk" → "talked")
        present_participle = verb + 'ing'  # Regular case (e.g., "talk" → "talking")
    
    # Third-person singular present: add "s" or "es"
    if verb[-1] in ['s', 'x', 'z'] or verb[-2:] in ['sh', 'ch']:
        third_person = verb + 'es'  # For verbs ending in 's', 'sh', 'ch', 'x', or 'z' (e.g., "pass" → "passes")
    elif verb[-1]=='y':
        third_person = verb[:-1] + 'ies'
    else:
        third_person = verb + 's'  # Regular case (e.g., "run" → "runs")

    
    return [(verb,'VERB'),(past_tense,'VERB_Past'),(present_participle,'VERB_PresC'),(third_person,'VERB_PresT')]

def deal_with_last_letter(word):
    if len(word)<3:
        return word
    if word[-1]==word[-2]: #run running, bed bedded 
        word=word[:-1]
    elif word[-1]=='i': #study studied
        word=word[:-1]+'y'
    elif word[-1] not in 'aeiouy': #love loving, note noted
        word=word+'e'
    return word

def get_base_form(init_word,pos,spell):
    word=init_word
    #remove the common prefix (-ed,-s,-ing,-er,-est)
    if word[-3:]=='ing' and pos=='VERB_PresC':
        word=word[:-3]
    elif word[-2:]=='ed' and pos=='VERB_Past':
        word=word[:-2]  

    elif pos in ['VERB_PresT','NOUN_P']:
        if word[-1:]=='s' and word[-2:]!='ss' and len(word)>3: #remove plural
            word=word[:-1]
            if word[-2:] in ['se', 'xe', 'ze'] or word[-3:] in ['she', 'che']:
                word=word[:-1]
            elif word[-2:]=='ie':
                word=word[:-2]+'y'
    
    if len(spell.known([word]))==0:
        word=deal_with_last_letter(word)
        if len(spell.known([word]))==0:
            #if the modified word does not belong to dictionnary anymore
            return init_word
    return word


def format_word(form,pos,spell):
    base_pos=pos.split('_')[0]
    assert base_pos in ['VERB','NOUN'],pos
    form=get_base_form(form,pos,spell)
    forms=[form]
    # no constraint on the number of letters
    if base_pos=='VERB':
        forms=vb_inflections(form,spell)
    elif base_pos=='NOUN':
        forms=nn_inflections(form)
    
    final_forms=[]
    for form,pos in forms:
        if len(spell.known([form]))>0 and len(form)>2:
            final_forms.append((form,pos)) 
    return final_forms



def space_characters(word,map_letters):
    #space symbols from letters and compute the symbol ratio
    new_word=''
    for i in range(len(word)):
        char=word[i]
        if char=='':
            continue
        elif char in map_letters['letters']:
            #current char is a letter
            new_word+=char
        else:
            #current char is not a letter
            if char not in map_letters['accepted_chars']:
                #char is an illegal character, skipping the whole word
                return None
            #char is a legal character but not a letter, we split it from the word
            new_word+=' '+char+' '

    #remove beginning and trailing char
    new_word=' '.join(list(filter(None,new_word.split(' '))))
    new_word=new_word.strip()
    return new_word

def format_pos(word,pos):
    if pos in ['NN', 'NNS']:    
        if word[-1]=='s':
            pos='NOUN_P'
        else:
            pos='NOUN'
    elif pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        if word[-2:]=='ed':
            pos='VERB_Past'
        elif word[-3:]=='ing':
            pos='VERB_PresC'
        elif word[-1]=='s':
            pos='VERB_PresT'
        else:
            pos='VERB' 
    else:
        pos='UNK'
    return pos



if __name__=='__main__':
    words =[('running','VERB_PresC'),('trees','NOUN_P'),('happiness','NOUN'),('induced','VERB_Past'),('induce','VERB')]
    #words=[('stable','ADJ')]
    spell = SpellChecker()
    for word in words:
        form,pos=word
        forms=format_word(form,pos,spell)
        print(form,':',forms)