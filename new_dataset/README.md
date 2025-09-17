**Vector corpus collection**

Repository for the new dataset on scientific articles related to vectors.

I- Metadata extraction

query_link : https://www.webofscience.com/wos/woscc/summary/2f398a71-d411-4f13-a114-bdac2864a1de-0172366f6b/relevance/1

The query is composed of one type of plant close to a type of insect closed to the notion of vector or transmission. The maximum distance between the plant and the insect and between the insect and the vector*/transmi* is 9 words. The following terms cannot appear in the results : malaria OR "west nile virus" OR dengue OR zika OR "human disease*" OR "human pathogen" OR "adverse-effects"OR ENCEPHAL* OR "transmission electron" OR "transposon vector" OR FILARIA* OR BLOOD OR SEAGRASS OR "chagas disease" OR BIT*

The query only search in WOS topic including Title, Abstract, Author Keywords and Keywords Plus


The 8,181 results were dowloaded on the 05/08/2025 in tsv format and correspond to the file "references_WOS.tsv" in query_V2 folder.


## Corpus construction steps

1. **Download full-text documents**  
   Use the script `extract_documents_from_source.py` to retrieve all documents.

2. **Retrieve metadata for articles citing the WOS query results**  
   Search by DOI using the script `get_references.py`:

   get_references.py --input_csv references_WOS.tsv --source coci
   get_references.py --input_csv references_WOS.tsv --source openalex

3. **Retrieve metadata for references cited by the WOS articles**  
   Use the script `get_info_ref.py`.

4. **Retrieve metadata for articles citing the references**  
   Use `get_references.py`:

   get_references.py --input_csv info_reference_openalex_dedup.tsv --source openalex
   get_references.py --input_csv info_reference_openalex_dedup.tsv --source coci
