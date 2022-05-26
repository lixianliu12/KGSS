import itertools
import spacy
import neuralcoref
import numpy as np
import re
import json
import argparse
from graph_show import GraphShow
#from geotext import GeoText
#from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens') #, device = 'cuda'
nlp = spacy.load("en_core_web_lg")
neuralcoref.add_to_pipe(nlp)
# list_of_relation = ['per:president', 'per:age','per:friendship','per:founder','per:affiliation', 'per:alumni_of', 'per:birth_date', 'per:birth_place', 'per:children', 'per:colleague', 'per:death_date', 'per:death_place', 'per:follows', 'per:funder', 'per:member_of', 'per:nationality', 'per:parent', 'per:sibling', 'per:sponsor', 'per:spouse', 'per:works_for', 'org:almuni', 'org:brand', 'org:dissolution_date', 'org:employee', 'org:founder', 'org:founding_date', 'org:founding_location', 'org:funder','org:location','org:member','org:sub_organization', 'loc:contained_in_place']
# per_list_of_relation = [i for i in list_of_relation if i.split(":")[0] == 'per']
# org_list_of_relation = [i for i in list_of_relation if i.split(":")[0] == 'org']
# loc_list_of_relation = [i for i in list_of_relation if i.split(":")[0] == 'loc']

#nlp.add_pipe(nlp.create_pipe('merge_noun_chunks'))

def sentenize(txt_path):
    # with open(txt_path, 'r') as f:
    #     f = f.read()
    #     f = nlp(f)._.coref_resolved
    #     print(f)
    #     text = re.sub(r'\n+', '.', f)
    #     text = re.sub(r'\[\d+\]', ' ', f)
    #     f = [str(i) for i in nlp(f).sents]
    #     return f
    
    f = nlp(txt_path)._.coref_resolved
    print(f)
    text = re.sub(r'\n+', '.', f)
    text = re.sub(r'\[\d+\]', ' ', f)
    f = [str(i) for i in nlp(f).sents]
    return f

def ent_filter(pair_list):
    ###################
    ## Filter Entity Pair
    ## Rule 1: Ent1 can be Person, organization only
    ##         if ent1 is a location, then ent2 must be a location
    ## Rule 2: For PERSON or ORG ent1, if the position of ent2 is higher than the ent1, we remove it
    ###################
    ### Spacy NER labels:
    ### CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, 
    ### ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART
    
    for pair in pair_list:
        if len(pair) < 3:
            if pair[0].split("@")[1] != "PERSON" and pair[0].split("@")[1] != "ORG" and pair[0].split("@")[1] != "GPE" and pair[0].split("@")[1] != "LOC":
                pair[0] = 'REMOVE'
            if pair[0] != 'REMOVE':
                if (pair[0].split("@")[1] == "GPE" or pair[0].split("@")[1] == "LOC") and (pair[1].split("@")[1] != "GPE" and pair[1].split("@")[1] != "LOC"):
                    pair[0] = 'REMOVE'
            if pair[0] != 'REMOVE':
                if pair[0].split("@")[1] == "PERSON" or pair[0].split("@")[1] == "ORG" or pair[0].split("@")[1] == "GPE" or pair[0].split("@")[1] == "LOC":
                    if (int(pair[0].split("@")[2]) > int(pair[1].split("@")[2])) and len(pair) < 3:
                        pair[0] = 'REMOVE'
                if pair[0].split("@")[0] == pair[1].split("@")[0]:
                    pair[0] = 'REMOVE'
    return pair_list    

def cosine(u, v):
            return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def sematic_similarity(sent_a, is_person, is_loc, list_of_relation):
    per_list_of_relation = [i for i in list_of_relation if i.split(":")[0] == 'per']
    org_list_of_relation = [i for i in list_of_relation if i.split(":")[0] == 'org']
    loc_list_of_relation = [i for i in list_of_relation if i.split(":")[0] == 'loc']
    sentence_embeddings = sbert_model.encode(sent_a)
    relation_dict = {}
    sim_list = []
    for i in range(len(sentence_embeddings)):
        if i < len(sentence_embeddings) - 1:
            sim = cosine(sentence_embeddings[0], sentence_embeddings[i + 1])
            sim_list.append(sim)
        else:
            break
    k = 0
    ####### FOR TAC ONLY #########
    if is_person:
        list_of_relation = per_list_of_relation
    elif is_loc:
        list_of_relation = loc_list_of_relation
    else:
        list_of_relation = org_list_of_relation
    ##############################
    for i in list_of_relation:
        relation_dict[i] = sim_list[k]
        k = k + 1
    return relation_dict

def sbert_similarity(ent, loc, country, sentence, relation):
    sentence_list = []
    start_ent = ent[0].split("@")[0]
    end_ent = ent[1].split("@")[0]
    if loc == True:
        if country == True:
            sent = start_ent + " is the country of " + end_ent
        else:
            sent = start_ent + " is located in " + end_ent
    elif len(ent) < 3:
        if int(ent[0].split("@")[2]) > int(ent[1].split("@")[2]):
            sent = sentence[sentence.find(end_ent) : sentence.find(start_ent) + len(start_ent)]
        else:
            sent = sentence[sentence.find(start_ent) : sentence.find(end_ent) + len(end_ent)]
    else:
        sent = start_ent + " " + ent[2] + " " + end_ent
    sentence_list.append(sent)

    ### relation schema ###
    list_of_relation = relation
    per_list_of_relation = [i for i in list_of_relation if i.split(":")[0] == 'per']
    org_list_of_relation = [i for i in list_of_relation if i.split(":")[0] == 'org']
    loc_list_of_relation = [i for i in list_of_relation if i.split(":")[0] == 'loc']
    #### This condition only applies for TAC dataset ###
    if ent[0].split("@")[1] == 'PERSON': #or ent[0].split("@")[1] == 'PRON':
        list_of_relation = per_list_of_relation
        person = True
        loc = False
    elif ent[0].split("@")[1] == 'GPE' or ent[0].split("@")[1] == 'LOC':
        list_of_relation = loc_list_of_relation
        person = False
        loc = True
    else: # Organization
        list_of_relation = org_list_of_relation
        person = False
        loc = False
    ####################################################
    
    for i in list_of_relation:
        sentence_list.append(ent[0].split("@")[0] + " " + i.split(":")[1].replace("_", " ") + " " + ent[1].split("@")[0])
    
    relation_dict = sematic_similarity(sentence_list, person, loc, list_of_relation)

    sorted_values = sorted(relation_dict.values(), reverse = True)
    sorted_dict = {}
    for i in sorted_values:
        for k in relation_dict.keys():
            if relation_dict[k] == i:
                sorted_dict[k] = relation_dict[k]
                break
    if sorted_dict[list(sorted_dict.keys())[0]] >= 0.8:
        #print(sorted_dict[list(sorted_dict.keys())[0]])
        return list(sorted_dict.keys())[0] #sorting the similarity value and return the one with the highest score
    else:
        return "no_relation"

def triple_extraction(f, is_noun, relation):
    events = []
    for tac_data in f:
        sentence = tac_data.replace("@", "").replace("\n","").replace('"',"").replace("(", "").replace(")", "")
        
        ###################
        ## NER detection
        ###################
        doc = nlp(sentence)
        ent_list = []

        ### find NERs and its position
        for ent in doc.ents:
            ent_list.append(ent.text.replace("'s", "").strip() + "@" + ent.label_ + "@" + str(ent.start))

        ### if noun selected, NOUN can be entity ###
        if is_noun:
            for token in doc:
                if token.pos_ == 'NOUN':
                    ent_list.append(token.text.replace("'s", "").strip() + "@" + token.pos_ + "@" + str(token.i))
                
        ###################
        ## Entity Permutation (construct entity pair)
        ###################
        pair_list = []
        for pair in (itertools.permutations(ent_list, 2)):
            pair_list.append(list(pair))

        ###################
        ## Relation Extraction based on rule-based template
        ## Rule 1 -- Noun + NER:PERSON ==> PERSON, job_title, Noun
        ## Rule 2 -- LOC + ',' + LOC   ==> LOC, is_part_of, LOC
        ## Rule 3 -- PER + ',' + NOUN  ==> PERSON, job_title, Noun
        ###################
        for pair in pair_list:
            ## Rule 1
            if pair[0].split("@")[1] == "PERSON":
                if int(pair[0].split("@")[2]) == int(pair[1].split("@")[2]) + 1:
                    pair.append('job_title')

            ## Rule 2
            if pair[0].split("@")[1] == "GPE" or pair[0].split("@")[1] == "LOC":
                subsent = sentence[sentence.find(pair[0].split("@")[0]):sentence.find(pair[1].split("@")[0]) + len(pair[1].split("@")[0])]
                result = re.search(pair[0].split("@")[0]+'(.*)'+pair[1].split("@")[0], subsent)
                if result:
                    if result.group(1).strip() == ',':
                        pair.append('is_part_of')

            ## Rule 3
            if pair[0].split("@")[1] == "PERSON":
                subsent = sentence[sentence.find(pair[0].split("@")[0]):sentence.find(pair[1].split("@")[0]) + len(pair[1].split("@")[0])]
                result = re.search(pair[0].split("@")[0]+'(.*)'+pair[1].split("@")[0], subsent)
                if result:
                    if result.group(1).strip() == ',':
                        pair.append('job_title')
        
        ### entity pair filtering
        pair_list = ent_filter(pair_list)
        pair_list_clean = [e for e in pair_list if e[0] != 'REMOVE']

        ### relation extraction over SBERT
        ###################
        ## Sentence-BERT relation classification
        ## 1. construct all pair of entity into a sentence
        ## 2. construct all predefined relation types given a pair of entities into a sentence
        ## 3. calculate the cosine similarity based on sentence from point 1 and point 2
        ###################
        for pair in pair_list_clean:
            if len(pair) < 3:
                pair.append(sbert_similarity(pair, False, False, sentence, relation))
            else:
                pair[2] = sbert_similarity(pair, False, False, sentence, relation)

        pair_list_clean = [e for e in pair_list_clean if e[2] != 'no_relation']
        
        for p in pair_list_clean:
            e = []
            e.append(p[0].split("@")[0])
            e.append(p[2].split(":")[1])
            e.append(p[1].split("@")[0])
            events.append(e)

    events = [list(x) for x in set(tuple(x) for x in events)]
    print(events)
    return events

### read in relation schema ###
def relation_list(path):
    with open(path) as f:
        list_relation = f.readlines()
    list_of_relation = [i.replace('\n', '') for i in list_relation]
    return list_of_relation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="input text")
    parser.add_argument("noun", help="select noun")
    parser.add_argument("relation", help="optional relation list")
    args = parser.parse_args()
    
    f = sentenize(args.text)

    if args.relation == 'NONE':
        relation = relation_list('./tacred_relation.txt')
    else:
        relation = relation_list('./upload/relation.txt')

    is_noun = False
    if args.noun == 'NOUN':
        is_noun = True
    events = triple_extraction(f, is_noun, relation)

    graph = GraphShow()
    graph.create_page(events, args.text)