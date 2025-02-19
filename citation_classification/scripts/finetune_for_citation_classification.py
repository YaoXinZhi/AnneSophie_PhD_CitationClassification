import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from get_citation_sequence import get_data_list
from citation_classifier import CitationClassifier, training_step, validation_step

import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm
import numpy as np
import random
from utils import CitationDataset, list_idx, plot_metric_evolution


    
def main(model_short_name, window, data_repartition, training_data_path):
  SEED = 42
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

  print("La version de torch est : ",torch.__version__)

  if torch.cuda.is_available():
      device = torch.device("cuda") 
      print(f"PyTorch is using GPU: {torch.cuda.get_device_name(0)}")
  else:
      device = torch.device("cpu")
 

  models = {'PubMedBERT':'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext','BioLinkBERT': 'michiyasunaga/BioLinkBERT-base','BioBERT': 'dmis-lab/biobert-v1.1', 'SciBERT': 'allenai/scibert_scivocab_uncased', 'RoBERTa-large': 'all-roberta-large-v1', 'RoBERTa' : 'roberta-base'}
  
  model_name = models[model_short_name]
  model = AutoModel.from_pretrained(model_name)
  print(f'Model {model} loaded to device')
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  tokenizer.add_tokens(['CITSEG'], special_tokens=True)
  model.resize_token_embeddings(len(tokenizer))
  
  num_of_epochs = 20
  lr = 2e-5
  ACC_STEP = 1
  batch_size = 32
  in_features = model.config.hidden_size
  hidden_layers = in_features
  
  citation_sequence_x_100citations, citation_sequence_y_100citations, citation_sequence_x_jiang, citation_sequence_y_jiang = get_data_list(window, training_data_path)
  all_labels = ['similar', 'neutral', 'usage', 'cocores', 'motivation', 'basis', 'weakness', 'future', 'support', 'cocogm', 'cocoxy']
  label_mapping = {label: idx for idx, label in enumerate(all_labels)}

  data_to_shuffle = list(zip(citation_sequence_x_jiang, citation_sequence_y_jiang))
  random.shuffle(data_to_shuffle)
  citation_sequence_x_jiang_shuffled, citation_sequence_y_jiang_shuffled = zip(*data_to_shuffle)
  citation_sequence_x_jiang_shuffled = list(citation_sequence_x_jiang_shuffled)
  citation_sequence_y_jiang_shuffled = list(citation_sequence_y_jiang_shuffled)

  
      
  if data_repartition == 'Jiang_train-PD_test':
    train_ratio = 0.80
    val_ratio = 0.20
    
    total_samples = len(citation_sequence_x_jiang_shuffled)
    train_size = int(train_ratio * total_samples)
    val_size = total_samples - train_size

    x_train, y_train = citation_sequence_x_jiang, citation_sequence_y_jiang
    x_train, y_train = citation_sequence_x_jiang_shuffled[:train_size], citation_sequence_y_jiang_shuffled[:train_size]
    x_val, y_val = citation_sequence_x_jiang_shuffled[train_size:train_size+val_size], citation_sequence_y_jiang_shuffled[train_size:train_size+val_size]

    x_test, y_test = citation_sequence_x_100citations, citation_sequence_y_100citations

  
    
  labels_idx_train, labels_idx_val, labels_idx_test = list_idx(label_mapping, y_train), list_idx(label_mapping, y_val), list_idx(label_mapping, y_test)

  citation_model = CitationClassifier(linear_size=hidden_layers, model=model, tokenizer=tokenizer, in_features=in_features, num_class=len(all_labels))
  citation_model.to(device)
  
  train_dataset = CitationDataset(text_citations=x_train, labels_ind=labels_idx_train, tokenizer=tokenizer, device=device)
  val_dataset = CitationDataset(text_citations=x_val, labels_ind=labels_idx_val, tokenizer=tokenizer, device=device)
  test_dataset = CitationDataset(text_citations=x_test, labels_ind=labels_idx_test, tokenizer=tokenizer, device=device)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  
  optimizer = AdamW(citation_model.parameters(), lr=lr, weight_decay= 0.002)
  loss_fn = nn.CrossEntropyLoss()
  num_training_steps = (len(train_loader) /ACC_STEP)
  total_steps = len(train_loader) * num_of_epochs
  warmup=0.2
  warmup_steps = math.floor(total_steps * warmup)
  scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

  tqdm.pandas()
  best_f1, best_macrof1, best_epoch  = 0, 0, 0
  train_losses, val_losses = [], []
  train_accuracies, val_accuracies = [], []
  val_F_measures, train_F = [], []
  val_macros, train_macros = [], []

  dic_values_per_class = {classe: {'F1': {'train': [], 'val': []}, 'P': {'train': [], 'val': []}, 'R': {'train': [], 'val': []}} for classe in all_labels}

  for i in tqdm(range(num_of_epochs)):          
    train_loss = training_step(train_loader, citation_model,optimizer, loss_fn, device, ACC_STEP, scheduler)
    train_acc, train_f1, train_loss2, train_F1macro, train_precision_per_class, train_recall_per_class, train_f1_per_class = validation_step(train_loader, citation_model, loss_fn, device, all_labels)
    val_acc, val_f1, val_loss, val_F1macro, val_precision_per_class, val_recall_per_class, val_f1_per_class = validation_step(val_loader, citation_model, loss_fn, device, all_labels)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    val_F_measures.append(val_f1)
    train_F.append(train_f1)
    val_losses.append(val_loss)
    val_macros.append(val_F1macro)
    train_macros.append(train_F1macro)

    for i, classe in enumerate(all_labels):
      dic_values_per_class[classe]['F1']['val'].append(val_f1_per_class[i])
      dic_values_per_class[classe]['P']['val'].append(val_precision_per_class[i])
      dic_values_per_class[classe]['R']['val'].append(val_recall_per_class[i])
      
    for i, classe in enumerate(all_labels):
      dic_values_per_class[classe]['F1']['train'].append(train_f1_per_class[i])
      dic_values_per_class[classe]['P']['train'].append(train_precision_per_class[i])
      dic_values_per_class[classe]['R']['train'].append(train_recall_per_class[i])


    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch=i

    if val_F1macro > best_macrof1:
        best_macrof1 = val_F1macro

    import pandas as pd
    mean_f1_per_class = {classe: np.nanmean(dic_values_per_class[classe]['F1']['val']) for classe in all_labels}
    print("Mean F1 par classe :", mean_f1_per_class )


    rows = []
    for class_name, metrics in dic_values_per_class.items():
        for metric, datasets in metrics.items():
            row = {'Class': class_name,'Metric': metric,'Train': str(datasets['train']),'Validation': str(datasets['val'])}
            rows.append(row)

    df = pd.DataFrame(rows)

    rows_mean = []
    for class_name, mean_f1 in mean_f1_per_class.items():
        rows_mean.append({'Class': class_name, 'Metric': 'Mean_F1_over_training', 'Train': '-', 'Validation': mean_f1})
    rows_mean.append({'Class': '-', 'Metric': 'Best_F1macro', 'Train': '-', 'Validation': best_macrof1})
    rows_mean.append({'Class': '-', 'Metric': 'Best_Val_F1', 'Train': '-', 'Validation': best_f1})


    df_means = pd.DataFrame(rows_mean)
    df = pd.concat([df, df_means], ignore_index=True)
    


    df.to_csv(f'metrics_per_class_{model_short_name}_{lr}_{ACC_STEP}_{data_repartition}_{window}.csv', index=False)


  print(f"{model_short_name} Best val F : {best_f1} for epoch {best_epoch} for LR {lr} for ACC_STEPS {ACC_STEP}")
  print(f"{model_short_name} Best val Fmacro : {best_macrof1} for epoch {best_epoch} for LR {lr} for ACC_STEPS {ACC_STEP}")

  plot_metric_evolution("Loss", train_losses, val_losses, num_of_epochs, model_short_name, lr, ACC_STEP, window)
  plot_metric_evolution("Accuracy", train_accuracies, val_accuracies, num_of_epochs, model_short_name, lr, ACC_STEP, window)
  plot_metric_evolution("F-score", train_F, val_F_measures, num_of_epochs, model_short_name, lr, ACC_STEP, window)
  plot_metric_evolution("Fmacro-score", train_macros, val_macros, num_of_epochs, model_short_name, lr, ACC_STEP, window)

  for cls, metrics in dic_values_per_class.items():
    for metric, repartitions in metrics.items():
      if metric == 'F1':
        train_values = repartitions['train']
        val_values = repartitions['val']
        plot_metric_evolution(f"{cls}_{metric}", train_values, val_values, num_of_epochs, model_short_name, lr, ACC_STEP, window)

    
  final_model_name = f"{model_short_name}_FINALMODEL_{lr}_accseteps{ACC_STEP}_ctx_{window}.pt"
  torch.save(citation_model, final_model_name)
  loaded_model = torch.load(final_model_name)
  loaded_model.eval()

  import pandas as pd
  csv_output = []
  predictions = []
  true_labels = []
  test_texts = []
  second_classes = []
  third_classes = []

  with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        citation_text = batch['citation_text']
          
        preds = loaded_model(tokens=input_ids, attention_mask=attention_mask)

        topk_values, topk_indices = torch.topk(preds, k=3, dim=-1)

        predicted_classes = torch.argmax(preds, dim=1).cpu().detach().numpy()
        second_class = topk_indices[:, 1].cpu().numpy()
        third_class = topk_indices[:, 2].cpu().numpy() 
          
        predictions.extend(predicted_classes)
        second_classes.extend(second_class)
        third_classes.extend(third_class)
        true_labels.extend(labels.cpu().numpy())
        test_texts.extend(citation_text)
  
  for i in range(len(true_labels)):
        csv_output.append({
            "Citation indice": i,
            "Citation text": test_texts[i],
            "Top1 Classe": all_labels[predictions[i]],
            "Top2 Classe": all_labels[second_classes[i]],
            "Top3 Classe": all_labels[third_classes[i]],
            "True Label": all_labels[true_labels[i]],
        })


  df = pd.DataFrame(csv_output)
  output_csv_path = f"{model_short_name}_predictions_{lr}_acc_steps_{ACC_STEP}_repartition_{data_repartition}_ctx_{window}.csv"
  df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Choose the model to vectorize citations")
  parser.add_argument(
        "--model",
        type=str,
        default="SciBERT",
        choices=["SciBERT", "BioBERT", "RoBERTa", "BioLinkBERT", "PubMedBERT"],
        help="The model to use. Options: 'SciBERT' (default), 'BioBERT', or 'RoBERTa'."
    )
  parser.add_argument(
        "--window_context",
        type=str,
        default='3-3',
        help="The window for the right and left context. Example of format : '2-3' for 2 sentences on the left and 3 sentences on the right, or None"
    )

  parser.add_argument(
        "--data_repartition",
        type=str,
        default="Jiang_train-PD_test",
        help="Data repartition : here we are training on the corpus Jiang2021 and testing on the PD corpus"
    )
  
  
  
  args = parser.parse_args()
  training_data_path = 'PATH_TO_TRAINING_DATA'

  main(args.model, args.window_context, args.data_repartition, training_data_path)