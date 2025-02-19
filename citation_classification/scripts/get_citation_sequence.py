import numpy as np
import pandas as pd
import re
import json
import argparse
import glob
   

def load_jiang_data(length_left, length_right, Jiang_data_path):
    def get_jiang_context(whole_context, ctx_length):
        return [whole_context[i].get("text") for i in range(min(len(whole_context), ctx_length))]

    citation_sequence_x_jiang, citation_sequence_y_jiang = [], []
    dic_classes = {}

    for f in glob.glob(Jiang_data_path):
        classname = f.split('/')[-1].split('.')[0]
        dic_classes[classname] = {'citance': [], 'left_ctx': [], 'right_ctx': [], 'sequence':[]}

        with open(f, 'r', encoding='utf-8') as json_file:
            content = json.load(json_file)
            
            for article_id, article_data in content.items():
                for ctx in article_data.get("citation_contexts", []):
                    text_pp = ctx.get("citance", {}).get("text_pp")

                    left_sentences = get_jiang_context(ctx.get("left_ctx", []), length_left)
                    right_sentences = get_jiang_context(ctx.get("right_ctx", []), length_right)

                    CITSEG_nb = 0
                    for token in text_pp.split(" "):
                        if "CITSEG" in token:
                            CITSEG_nb +=1

                    if CITSEG_nb < 2:
                        #we only want to classify one ref
                        dic_classes[classname]['citance'].append(text_pp)
                        dic_classes[classname]['left_ctx'].append(left_sentences)
                        dic_classes[classname]['right_ctx'].append(right_sentences)
                        sequence = "".join(left_sentences) + text_pp + "".join(right_sentences)
                        dic_classes[classname]['sequence'].append(sequence)
                    else:
                        pass
                        

    for classname, dic_text in dic_classes.items():
       for seq in dic_text["sequence"]:
          citation_sequence_y_jiang.append(classname)
          citation_sequence_x_jiang.append(seq)

    return citation_sequence_x_jiang, citation_sequence_y_jiang





def load_PD_data(length_left, length_right):
    def clean_sentence(sentence):
        cleaned_sentence = re.sub(r'<ref[^>]*>', '', sentence)
        cleaned_sentence = re.sub(r'type="[^"]*"\s*target="[^"]*">|type="bibr">', '', cleaned_sentence)
        cleaned_sentence = re.sub(r'</ref>|<ref', '', cleaned_sentence)
        return cleaned_sentence
    
    def define_y_100citation(labels):
        y = []
        for i in range(len(labels)):
            label = labels[i].split('|')[0].replace(' ', '').lower()
            y.append(label)
        return y
    
    def load_context_and_citances(df, begining_citation_sentences, end_citation_sentences, length_left, length_right):
        citances = []
        left_context_sentences = []
        right_context_sentences = []
        context_dic_lists = {}

        for i in range(len(begining_citation_sentences)):
            citance = begining_citation_sentences[i]+ ' (CITSEG) '+end_citation_sentences[i]
            citance = clean_sentence(citance)
            citances.append(citance)

            if length_left is not None and length_right is not None:

                left_context = ''
                right_context = ''

                for n in range(1, length_left +1):
                    context_dic_lists[f'l{n}'] = df[f"l{n}"].astype(str).tolist()

                for n in range(1, length_right + 1):
                    context_dic_lists[f'r{n}'] = df[f"r{n}"].astype(str).tolist()

                for n in reversed(range(1, length_left+ 1)):
                    left_context += context_dic_lists[f'l{n}'][i] if context_dic_lists[f'l{n}'][i] != 'nan' else ''
                left_context = clean_sentence(left_context)
                left_context_sentences.append(left_context)


                for n in range(1, length_right +1):
                    right_context += context_dic_lists[f'r{n}'][i]+' ' if context_dic_lists[f'r{n}'][i] != 'nan' else ''
                    right_context = clean_sentence(right_context)
                right_context_sentences.append(right_context)
        
        return left_context_sentences, right_context_sentences, citances

    dataset ='../../Datasets/100_citation_sample.csv'
    df = pd.read_csv(dataset)
    begining_citation_sentences = df["citation_sentence"].astype(str).tolist()
    end_citation_sentences = df["end_citation_sentence"].astype(str).tolist()

    labels = pd.read_csv('../../Datasets/100_citation_sample.csv')["annotation_rhetorical_function"].astype(str).tolist()

    citation_sequence_y_100citations = define_y_100citation(labels)
    left_context_sentences, right_context_sentences, citances = load_context_and_citances(df, begining_citation_sentences, end_citation_sentences, length_left, length_right)
    citation_sequence_x_100citations = [left_context_sentences[i]+citances[i] + right_context_sentences[i] for i in range(len(citances))]

    return citation_sequence_x_100citations, citation_sequence_y_100citations



def to_categorical(labels, all_labels):
    return [all_labels.index(lbl.lower()) for lbl in labels]

def get_data_list(window_context, Jiang_data_path):
  if window_context is not None:
    length_left, length_right = int(window_context.split('-')[0]), int(window_context.split('-')[1])
  else:
     length_left, length_right = 0, 0

  citation_sequence_x_100citations, citation_sequence_y_100citations = load_PD_data(length_left, length_right)
  citation_sequence_x_jiang, citation_sequence_y_jiang = load_jiang_data(length_left, length_right, Jiang_data_path)
  return citation_sequence_x_100citations, citation_sequence_y_100citations, citation_sequence_x_jiang, citation_sequence_y_jiang

