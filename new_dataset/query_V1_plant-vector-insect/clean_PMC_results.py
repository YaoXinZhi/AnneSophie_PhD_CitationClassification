import pandas as pd
import json
from unidecode import unidecode

def is_human_disease(human_disease_terms, title):
    is_human_disease = False
            
    for term in human_disease_terms:
        if term in title:
            is_human_disease = True
            break
    if "zika" in title or "dengue" in title or "west nile" in title:
        is_human_disease = True
    return is_human_disease

def is_stop_word_title(stopwords_title, title):
    if len(title.split(" ")) < 5:
        return True
    for word in stopwords_title:
        if word in title:
            print(f"Found stopword : {word}")
            return True
    return False

def is_stop_journal(stop_journals, journal):
    for jname in stop_journals:
        if jname in journal:
            print(f"Found stopword : {journal}")
            return True
    return False

input_file = "metadata_deduplicated.jsonl"
output_file = "filtered_metadata.jsonl"
human_disease_list = "sorted_human_disease.lst"
deleted_lines_file = "deleted_data.jsonl"
title_stopwords_path = 'title_stopwords.lst'

# human disease lexique ref : https://doi.org/10.6084/m9.figshare.1367443.v1
with open(human_disease_list, "r") as lexique:
    human_disease_terms = [line.strip().lower() for line in lexique]

#list of stopwords for titles to be completed - possibility to do a list of stopwords for abstracts
with open(title_stopwords_path, "r") as stopwords_title_f:
    stopwords_titles = [line.strip().lower() for line in stopwords_title_f]

#list of stopwords journals
with open('stop_journals.lst', "r") as stopwords_journals_f:
    stop_journals = [line.strip().lower() for line in stopwords_journals_f]


with open(input_file, "r") as infile, open(output_file, "w") as outfile, open(deleted_lines_file, "w") as deleted_file:
    for line in infile:
        data = json.loads(line)
        title = unidecode(data.get("title", "").lower())
        journal = unidecode(data.get("journal", "").lower())
        #print(title)

        title = unidecode(title.lower())

        if is_stop_word_title(stopwords_titles, title):
            print(f"\nDeleting (stopword or too short): {title}")
            deleted_file.write(line)
        elif is_human_disease(human_disease_terms, title):
            deleted_file.write(line)
        elif is_stop_journal(stop_journals, journal):
            deleted_file.write(line)
        else:
            outfile.write(line)

print(f"Filtered file saved as {output_file}")
print(f"Deleted lines saved as {deleted_lines_file}")
