import numpy as np
import pandas as pd
import re
import json
import argparse
import glob
   
def get_indices_from_delimiter(sequence, delimiter='@'):
        """Retourne les indices de chaque partie de la séquence délimitée par un séparateur."""
        parts = sequence.split(delimiter) 
        indices = []
        current_pos = 0
        
        for part in parts:
            part_indices = [] 
            for word in part.split():
                part_indices.append((current_pos, current_pos + len(word)))
                current_pos += len(word) + 1 
            indices.append(part_indices)
    
        return indices

def load_jiang_data(length_left, length_right):
    def get_jiang_context(whole_context, ctx_length):
        return [whole_context[i].get("text").replace("@", "") for i in range(min(len(whole_context), ctx_length))]

    citation_sequence_x_jiang = []
    citation_sequence_y_jiang = []
    jiang_left_contexts = []
    list_indices = []
    jiang_right_contexts = []
    jiang_citances = []
    dic_classes = {}

    for f in glob.glob('path_to_Jiang2021_data/*.json'):
        classname = f.split('/')[-1].split('.')[0]
        dic_classes[classname] = {'citance': [], 'left_ctx': [], 'right_ctx': [], 'sequence':[], 'indice':[]}

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
                        dic_classes[classname]['citance'].append(text_pp)
                        dic_classes[classname]['left_ctx'].append(left_sentences)
                        dic_classes[classname]['right_ctx'].append(right_sentences)
                        sequence = "".join(left_sentences) + text_pp + "".join(right_sentences)
                        
                        sequence = "".join(left_sentences) + " @ " + text_pp + " @ " + "".join(right_sentences)

                        context_indices = [get_indices_from_delimiter(citation) for citation in sequence]

                        dic_classes[classname]['sequence'].append(sequence)
                        dic_classes[classname]['indice'].append(context_indices)
                    else:
                        pass
                        

    for classname, dic_text in dic_classes.items():
       for seq in dic_text["sequence"]:
          citation_sequence_y_jiang.append(classname)
          citation_sequence_x_jiang.append(seq)
       for seq in dic_text["citance"]:
          jiang_citances.append(seq)
       for seq in dic_text["left_ctx"]:
          jiang_left_contexts.append(seq)
       for seq in dic_text["right_ctx"]:
          jiang_right_contexts.append(seq)
       for ind in dic_text["indice"]:
          list_indices.append(ind)

    return citation_sequence_x_jiang, citation_sequence_y_jiang, jiang_citances, jiang_left_contexts, jiang_right_contexts, list_indices

def get_data_list(window_context):
  if window_context is not None:
    length_left, length_right = int(window_context.split('-')[0]), int(window_context.split('-')[1])
  else:
     length_left, length_right = 0, 0
  citation_sequence_x_jiang, citation_sequence_y_jiang, jiang_citances, jiang_left_contexts, jiang_right_contexts, Jiang_list_indices = load_jiang_data(length_left, length_right)
 
  return citation_sequence_x_jiang, citation_sequence_y_jiang, jiang_citances, jiang_left_contexts, jiang_right_contexts, Jiang_list_indices

