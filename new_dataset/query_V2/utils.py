import os
import glob
import xml.etree.ElementTree as ET
from unidecode import unidecode


def get_downloaded_dois():
    '''Check if the DOI has already been dowloaded by searching in the repertories :
    in_PMC, elsevier_API_xml, Cross_ref_downloaded_pdfs, pdfs_unpaywall, Wiley_API, PLOS_API
    The directories have to be updated !
    Also now the pmc files will be saved with the doi name for more coherence, the function will be updated accordinlgy
    '''
    
    seen = set()
    
    for file in ["downloaded.lst"]:
        if os.path.exists(file):
            with open(file, "r") as f:
                seen.update(line.strip() for line in f)


    for pmcid_file in glob.glob("../in_PMC/*.xml"):
        tree = ET.parse(pmcid_file)
        root = tree.getroot()
        #get doi from pmc xml file
        doi_el = root.find('.//article-id[@pub-id-type="doi"]')
        doi = doi_el.text if doi_el is not None else None
        if doi:
            seen.add(normalize_doi(doi))

    
    seen.update(normalize_doi(doi.split("/")[-1].replace(".xml", "").replace("_", "/")) for doi in glob.glob("../elsevier_API_xml/*.xml"))
    seen.update(normalize_doi(f.split("/")[-1].replace(".pdf", "").replace("_", "/")) for f in glob.glob("../Cross_ref_downloaded_pdfs/*.pdf"))
    seen.update(normalize_doi(f.split("/")[-1].replace(".pdf", "").replace("_", "/")) for f in glob.glob("../pdfs_unpaywall/*.pdf"))
    seen.update(normalize_doi(f.split("/")[-1].replace(".pdf", "").replace("_", "/")) for f in glob.glob("../Wiley_API/*.pdf"))
    seen.update(normalize_doi(f.split("/")[-1].replace(".pdf", "").replace("_", "/")) for f in glob.glob("../PLOS_API/*.pdf"))
    seen.update(normalize_doi(f.split("/")[-1].replace(".xml", "").replace("_", "/")) for f in glob.glob("../PLOS_API/*.xml"))
    seen.update(normalize_doi(f.split("/")[-1].replace(".pdf", "").replace("_", "/").replace("DOI-", "")) for f in glob.glob("../in_ISTEX/*.pdf"))

    return seen

def normalize_doi(doi):
    '''normalize a doi for comparition by lowering, replace https, and removing final . and ,'''
    return unidecode(str(doi).lower().strip().replace("https://doi.org/","").rstrip(".").rstrip(","))