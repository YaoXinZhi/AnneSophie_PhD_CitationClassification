import requests
import pandas as pd
import csv
import json
import time
from utils import *
from tqdm import tqdm

import time
import os
import glob
import config_APIs
import argparse
import ast

def extract_selected_fields(work):

    authors = [author.get('author', {}).get('display_name', '') for author in work.get('authorships', [])]
    authors_str = "; ".join(authors) if authors else None

        
    return (
        work.get('id'),
        work.get('ids'),
        work.get('doi'),
        work.get('publication_year'),
        work.get('title'),
        authors_str,
        work.get('host_venue', {}).get('display_name'),
        work.get('host_venue', {}).get('publisher'),
        work.get('host_venue', {}).get('issn_l'),
        work.get('open_access', {}).get('is_oa'),
        work.get('cited_by_count'),
        len(work.get('referenced_works', [])),
        work.get('type_crossref'),
        work.get('type'),
        work.get('language'),
        work.get('indexed_in'),
        work.get('open_access'),
        work.get('is_retracted')

    )



def retrieve_citing_articles_COCI(doi):
    #new coci API version from february 2025
    url = f"https://api.opencitations.net/index/v2/citations/doi:{doi}"
    df_citing = pd.DataFrame()


    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()

            df_citing = pd.DataFrame(data)
            df_citing["Doi_Reference_Processed"] = doi
    except Exception as e:
        print(f"Erreur lors de la récupération des citations COCI pour {doi}: {e}")
    
    return df_citing 


def retrieve_citing_works_from_openAlex(doi, mailto="", max_results=1000):
    """
    Retrieves metadata of works that cite the article with the given DOI using OpenAlex API.
    """
    #info that we will save
    column_names = [
    'id',
    'ids',
    'doi',
    'publication_year',
    'title',
    'authors',
    'host_venue',
    'publisher',
    'journal_issn',
    'is_oa',
    'cited_by_count',
    'referenced_count',
    'type_crossref',
    'type',
    'language',
    'indexed_in',
    'open_access',
    'is_retracted']

    #Get OpenAlex ID from DOI
    headers = {"User-Agent": f"{mailto}"}
    r = requests.get(f"https://api.openalex.org/works/https://doi.org/{doi}", headers=headers)
    if r.status_code != 200:
        print(f"Error {r.status_code}")
        return pd.DataFrame()

    ref_json = r.json()
    openalex_id = r.json().get('id')

    if not openalex_id:
        return pd.DataFrame()

    #Fetch citing works filter=cites:<OpenAlex ID>
    citing_works = []
    per_page = 200
    cursor = "*"
    count = 0

    while count < max_results:
        url = (f"https://api.openalex.org/works?filter=cites:{openalex_id.replace('https://openalex.org/', '')}"
            f"&per-page={per_page}&cursor={cursor}&mailto={mailto}")

        r = requests.get(url, headers=headers)
        
        if r.status_code != 200:
            print(f"Failed to retrieve citing works: {r.status_code}")

            if r.status_code == 429:
                print("Too many requests (429). Stopping script.")
                return 'Time_to_stop'

            elif r.status_code == 503:
                time.sleep(1)

            break

        data = r.json()
        results = data.get("results", [])
        
        for work in results:
            citing_works.append(extract_selected_fields(work))
            count += 1
            if count >= max_results:
                break

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    df_citing = pd.DataFrame(citing_works, columns=column_names)

    return df_citing

def main(input_csv, source):
    '''Input is the metadata resulting of the WOS query named references_WOS.tsv or the informations about the references without duplicates info_reference_openalex_dedup.tsv
    Looks for the articles citing a DOI through COCI or openalex API'''

    if input_csv == "references_WOS.tsv":
        df = pd.read_csv(input_csv, sep="\t").fillna("")
        doi_list_to_process = df["DI"].tolist()
        if source == "coci":
            output_name = "COCI_references_citations.csv"
        elif source == "openalex":
            output_name = "OPENALEX_citing_references_citations.csv"
        else:
            print("Wrong source, please chose openalex or coci")

    elif input_csv == "info_reference_openalex_dedup.tsv":
        df_info_ref = pd.read_csv(input_csv, sep="\t").fillna("")
        df_articles = df_info_ref[df_info_ref["Reference_Type"] == "article"]
        doi_list_to_process = []

        #take the openalex doi ids
        for ids_row in df_articles["Reference_IDS"].tolist():
                ids_dict = ast.literal_eval(ids_row)
                doi_url = ids_dict.get("doi", "")
                if doi_url:
                    doi = doi_url.replace("https://doi.org/", "").strip().lower()
                    doi_list_to_process.append(doi)
        
        if source == "coci":
            output_name = "CoCi_other_citing_references.csv"

        elif source == "openalex":
            output_name = "Openalex_other_citing_references.csv"

        else:
            print("Wrong source, please chose openalex or coci 'Openalex_other_citing_references'")

    print(f"This script will create the output {output_name}")
    processed_ref = set()

    # If output already exists, reload processed DOIs
    if os.path.exists(output_name):
        df_existing_references = pd.read_csv(output_name, sep=",")
        processed_ref = set(normalize_doi(str(doi)) for doi in df_existing_references["Doi_Reference_Processed"].dropna().unique())

    for doi in tqdm(doi_list_to_process, desc="Retrieving citing articles"):
        doi = normalize_doi(doi)

        if not doi or str(doi).lower() == "nan":
            #no doi --> skip
            continue

        elif doi in processed_ref:
            #doi already processed --> skip
            continue

        if "Openalex" in output_name:
            df_citing_articles_open_alex = retrieve_citing_works_from_openAlex(doi, mailto=config_APIs.INSTITUTION_EMAIL)

            if df_citing_articles_open_alex is not None and not df_citing_articles_open_alex.empty:
                new_rows_openalex = []
                for _, row in df_citing_articles_open_alex.iterrows():
                    new_rows_openalex.append({
                        "Doi_Reference_Processed": doi,
                        "Citing_ID": row.get("id"),
                        "Citing_DOI": normalize_doi(row.get("doi")),
                        "Citing_IDS": row.get("ids"),
                        "Citing_Title": row.get("title"),
                        "Citing_Authors": row.get("authors"),
                        "Citing_Journal": row.get("journal"),
                        "Citing_Publisher": row.get("publisher"),
                        "Citing_Journal_ISSN": row.get("journal_issn"),
                        "Citing_Publication_Year": row.get("publication_year"),
                        "Citing_Is_OA": row.get("is_oa"),
                        "Citing_Citation_Count": row.get("cited_by_count"),
                        "Citing_Reference_Count": row.get("referenced_count"),
                        "Citing_Type_Crossref": row.get("type_crossref"),
                        "Citing_Type": row.get("type"),
                        "Citing_Language": row.get("language"),
                        "Citing_Indexed_In": row.get("indexed_in"),
                        "Citing_Open_Access_Details": row.get("open_access"),
                        "Citing_Is_Retracted": row.get("is_retracted"),
                    })

                df_new_openalex = pd.DataFrame(new_rows_openalex)
                df_new_openalex.to_csv(output_name, sep=",", index=False, mode="a", header=not os.path.exists(output_name))
            else:
                print(f"No citing works found for {doi} in OpenAlex")

        else:
            df_new_coci = retrieve_citing_articles_COCI(doi)

            if not df_new_coci.empty:
                df_new_coci.to_csv(output_name, sep=",", index=False, mode="a",header=not os.path.exists(output_name))
            else:
                print(f"No citing works found for {doi} in COCI")

            time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve citing articles from OpenAlex or COCI")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input TSV file containing DOIs")
    parser.add_argument("--source", type=str, choices=["openalex", "coci"], required=True, help="Choose data source: 'openalex' or 'coci'")

    args = parser.parse_args()
    main(args.input_csv, args.source)
