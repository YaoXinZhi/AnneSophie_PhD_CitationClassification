# Classification des fonctions rhétoriques des citations par discipline

Ce dépôt contient plusieurs fichiers relatifs à la classification des fonctions rhétoriques des citations à travers différentes disciplines.

## Contenu du dépôt

### `PD100cit`
Ce dossier contient :  
- **`100_citation_sample.csv`** : 100 citations tirées aléatoirement du corpus *Pear Decline* (PD100cit), avec leurs contextes et annotations.  
- **`annotation_guidelines.pdf`** : Guide d'annotation utilisé comme référence pour les annotations de PD100cit, avec des exemples en écologie.  

### `script`
Ce dossier contient plusieurs scripts pour la classification des citations :  

- **Scripts :**  
  - `finetune_for_citation_classification.py` : Script principal pour fine-tuner un modèle de langue (**BioBERT, SciBERT, RoBERTa, BioLinkBERT**), à définir en argument, avec la taille de la fenêtre de contexte.  
  - `citation_classifier.py` : Classifieur et fonctions d'entraînement et de validation.  
  - `get_citation_sequence.py` : Extraction des séquences de citations des deux corpus, sorties sous forme de listes.  
  - `utils.py` : Fonctions auxiliaires utilisées par `finetune_for_citation_classification.py`.  

- **Instruction pour le prompt de GPT-4** : Fichier Prompt_instructions_citation_classification.pdf

- **Sous-dossier `essai_modele_Jiang`**  
  - `fine_tune_several_vectors.py` : Script principal pour reproduire le modèle de **Jiang & Chen (2023)**.  
  - `citation_classifier.py` : Classifieur et fonctions d'entraînement et de validation, appelé par `fine_tune_several_vectors.py`.  
  - `prepare_data.py` : Extraction des séquences de citations et de leurs contextes, avec délimitation par `@`. Sorties sous forme de listes. Script appelé par `fine_tune_several_vectors.py`.  
