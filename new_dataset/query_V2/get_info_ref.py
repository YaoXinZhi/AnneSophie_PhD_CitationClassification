import pickle
import pandas as pd
from tqdm import tqdm
import os
import config_APIs
import re
import requests
from unidecode import unidecode
from utils import normalize_doi
from get_reference import extract_selected_fields


def get_WOS_ref(file_path):
    """
    Load WOS references file (TSV) and extract publication year, DOI, title, authors, and raw references
    """
    df = pd.read_csv(file_path, sep="\t").fillna("")

    articles = []

    for i, row in df.iterrows():
        article = {"title": row["TI"], "author": row["AU"], "doi": row["DI"], "year": row["PY"], "references": row["CR"]}

        articles.append(article)

    return articles

def retrieve_citing_works_from_openAlex(doi, mailto="", max_results=1000):
    """
    Retrieves metadata of works the given DOI using OpenAlex API.
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

    #Get information about the ref with openalex
    headers = {"User-Agent": f"{mailto}"}
    r = requests.get(f"https://api.openalex.org/works/https://doi.org/{doi}", headers=headers)

    if r.status_code != 200:
        print(f"ERROR while searching DOI {doi} status code {r.status_code}")
        return pd.DataFrame()

    ref_json = r.json()
    openalex_id = r.json().get('id')

    if not openalex_id:
        #not in openAlex
        return pd.DataFrame(), pd.DataFrame()


    reference_info = extract_selected_fields(ref_json)
    df_current_article = pd.DataFrame([reference_info], columns=column_names)

    return df_current_article

def extract_ref_metadata(ref):
    """
    Extrait le DOI, le titre (simplifié), les auteurs (simplifiés) et l'année d'une chaîne de référence brute.
    """
    #DOI pattern
    doi_match = re.search(r"10\.\d{4,9}/\S+", ref)
    ref_doi = doi_match.group(0) if doi_match else ""


    #Année (format: 4 chiffres, généralement entre parenthèses ou seuls)
    year_match = re.search(r"\b(19|20)\d{2}\b", ref)
    ref_year = year_match.group(0) if year_match else ""

    
    parts = ref.split(", ")
    ref_authors = parts[0].strip()
    if len(parts) >= 3:
        ref_title = parts[2].strip()
    else:
        ref_title = ''

    return {"doi": ref_doi, "year": ref_year, "authors": ref_authors, "title": ref_title }

def create_dedup():
    '''Function to deduplicate the references info, 
    the output is the input for get_reference --> get citing articles for the references'''

    info_reference_type_path = "info_reference_openalex.tsv"
    df_info_ref = pd.read_csv(info_reference_type_path, sep="\t").fillna("")
    df_info_ref["Reference_DOI"] = df_info_ref["Reference_DOI"].apply(normalize_doi)
    df_info_ref = df_info_ref.drop_duplicates(subset=["Reference_DOI"], keep="last")
    df_info_ref.to_csv("info_reference_openalex_dedup.tsv", sep="\t", index=False)

def extract_info_reference(output_reference_details="info_reference_openalex.tsv"):
    """
    Parcourt les références de la query WOS et cherche info complémentaires
    """
    #Charger les DOI déjà traités si le fichier existe
    processed_ref = set()

    if os.path.exists(output_reference_details):
        df_existing_references = pd.read_csv(output_reference_details, sep="\t")
        processed_ref = set(df_existing_references["Reference_DOI"].apply(normalize_doi))

    # Charger les articles query (WOS)
    WOS_references_path = "references_WOS.tsv"
    articles_query = get_WOS_ref(WOS_references_path)
    df = pd.read_csv(WOS_references_path, sep="\t").fillna("")

    references = df["CR"].tolist()
    set_doi_to_process = set()

    
    for article in tqdm(articles_query, desc="Processing WOS references"):
        if not article.get("references"):
            #if there are no references skip
            continue

        #list all the references for one article
        ref_list = [ref.strip() for ref in article["references"].split(";") if ref.strip()]

        if not ref_list:
            continue

        for i in range(len(ref_list)-1):

            if len(ref_list[i+1]) < 2 and int(ref_list[i+1]):
                    #sometimes there is a ; in the doi but i splitted with ";" so i cannot find those DOIs
                    #check if the next value in the list is a numeric and a single value to complete the doi
                    ref_doi = ref_doi + ";" + ref_list[i+1]

            ref = ref_list[i]
            ref_meta = extract_ref_metadata(ref)
            ref_doi = normalize_doi(ref_meta.get("doi"))
            ref_title = ref_meta.get("title")
            ref_authors = ref_meta.get("authors")
            ref_year = ref_meta.get("year")

            doi_candidates = []
            if "DOI [" in ref:
                doi_candidates = re.findall(r"10\.\d{4,9}/[^\s,\]]+", ref)
            
            if not ref_doi or ref_doi in processed_ref:
                continue

            elif ref_doi and ref_doi not in processed_ref:                    
                count_references_with_doi += 1
                count +=1
                
                df_citing_articles_open_alex, df_current_article = retrieve_citing_works_from_openAlex(ref_doi, mailto=config_APIs.INSTITUTION_EMAIL)

                if df_current_article.empty and doi_candidates:
                    #sometimes there are a list of dois and some are valid but the others not found by openalex
                    for i in range(len(doi_candidates)-1):
                        cand = doi_candidates[i+1]
                        df_citing_articles_open_alex, df_current_article = retrieve_citing_works_from_openAlex(cand, mailto=config_APIs.INSTITUTION_EMAIL)
                        if not df_current_article.empty:
                            break


                new_rows = []
                new_rows_ref = []
                if isinstance(df_citing_articles_open_alex, str) and df_citing_articles_open_alex == 'Time_to_stop':
                    break
                
                if df_current_article is not None:
                    for _, row in df_current_article.iterrows():
                        reference_article_id = row.get("id")
                        reference_article_doi = normalize_doi(row.get("doi"))
                        reference_article_ids = row.get("ids")
                        reference_article_title = row.get("title")
                        reference_article_authors = row.get("authors")
                        reference_article_journal = row.get("journal")
                        reference_article_publisher = row.get("publisher")
                        reference_article_journal_issn = row.get("journal_issn")
                        reference_article_year = row.get("publication_year")
                        reference_article_is_oa = row.get("is_oa")
                        reference_article_cited_by_count = row.get("cited_by_count")
                        reference_article_referenced_count = row.get("referenced_count")
                        reference_article_type_crossref = row.get("type_crossref")
                        reference_article_type = row.get("type")
                        reference_article_language = row.get("language")
                        reference_article_indexed_in = row.get("indexed_in")
                        reference_article_open_access = row.get("open_access")
                        reference_article_is_retracted = row.get("is_retracted")

                        new_rows_ref.append({
                        # Reference (the current article)
                        "Reference_ID": reference_article_id,
                        "Reference_DOI": ref_doi,
                        "Reference_IDS": reference_article_ids,
                        "Reference_Title": reference_article_title,
                        "Reference_Authors": reference_article_authors,
                        "Reference_Journal": reference_article_journal,
                        "Reference_Publisher": reference_article_publisher,
                        "Reference_Journal_ISSN": reference_article_publisher,
                        "Reference_Publication_Year": reference_article_year,
                        "Reference_Is_OA": reference_article_is_oa,
                        "Reference_Citation_Count": reference_article_cited_by_count,
                        "Reference_Reference_Count": reference_article_referenced_count,
                        "Reference_Type_Crossref": reference_article_type_crossref,
                        "Reference_Type": reference_article_type,
                        "Reference_Language": reference_article_language,
                        "Reference_Indexed_In": reference_article_indexed_in,
                        "Reference_Open_Access_Details": reference_article_open_access,
                        "Reference_Is_Retracted": reference_article_is_retracted})
                else:
                    new_rows_ref.append({
                        "Doi_Reference_Processed": ref_doi,
                        "Title_Reference_Processed": ref_title,
                        "Authors_Reference_Processed": ref_authors,
                        "Year_Reference_Processed": ref_year,
                        "Reference_ID": 'ERROR',
                        "Reference_DOI": 'ERROR',
                        "Reference_IDS": 'ERROR',
                        "Reference_Title": 'ERROR',
                        "Reference_Authors": 'ERROR',
                        "Reference_Journal": 'ERROR',
                        "Reference_Publisher": 'ERROR',
                        "Reference_Journal_ISSN": 'ERROR',
                        "Reference_Publication_Year": 'ERROR',
                        "Reference_Is_OA": 'ERROR',
                        "Reference_Citation_Count": 'ERROR',
                        "Reference_Reference_Count": 'ERROR',
                        "Reference_Type_Crossref": 'ERROR',
                        "Reference_Type": 'ERROR',
                        "Reference_Language": 'ERROR',
                        "Reference_Indexed_In": 'ERROR',
                        "Reference_Open_Access_Details": 'ERROR',
                        "Reference_Is_Retracted": 'ERROR'
                    })

                if new_rows_ref:
                    df_new = pd.DataFrame(new_rows_ref)
                    df_new.to_csv(output_reference_details, sep="\t", index=False, mode="a", header=not os.path.exists(output_reference_details))



if __name__ == "__main__":
    extract_info_reference()