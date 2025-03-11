import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from prepare_data import get_data_list
from citation_classifier import CitationClassifier, training_step, validation_step

import torch.nn as nn
import math
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import random



class CitationDataset(Dataset):
    def __init__(self, text_citations : list, labels_ind : list, tokenizer, citances, left_ctxs, right_ctxs, ctx_ind):
        self.text_citations = text_citations
        self.labels_ind = labels_ind
        self.max_token_len = 512
        self.tokenizer = tokenizer
        self.citances = citances
        self.left_ctxs = left_ctxs
        self.right_ctxs = right_ctxs
        self.ctx_inds= ctx_ind


    def __len__(self):
        return len(self.labels_ind)

    def __getitem__(self, idx):
        citation_text = self.text_citations[idx]
        
        citance = self.citances[idx]
        right_ctx = self.right_ctxs[idx]
        left_ctxs = self.left_ctxs[idx]
        label = self.labels_ind[idx]
        inputs = self.tokenizer(citation_text, return_tensors='pt', padding='max_length', max_length=self.max_token_len, truncation=True, add_special_tokens=True, return_attention_mask=True)

        citance = " @ " + citance + " @ "
        citance_tokens = self.tokenizer(citance, add_special_tokens=True)
        citance_length = len(citance_tokens["input_ids"])
        
        max_length = self.max_token_len
        nb_tokens = citance_length - 2

        new_sequence = citance
        i, j = 0, 0
        counter = 0
        max_iterations = len(right_ctx) + len(left_ctxs)

        while nb_tokens < 512 and counter < max_iterations:
            if nb_tokens >= 512:
               break
            
            if i < len(right_ctx):
                next_right_tokens = len(self.tokenizer(right_ctx[i], add_special_tokens=False)["input_ids"])
                if nb_tokens + next_right_tokens < 512:
                    new_sequence += " " + right_ctx[i]
                    nb_tokens += next_right_tokens
                    i += 1
                else:
                   break

            if j < len(left_ctxs):
                next_left_tokens = len(self.tokenizer(left_ctxs[j], add_special_tokens=False)["input_ids"])
                if nb_tokens + next_left_tokens < 512: 
                    new_sequence = left_ctxs[-j] + new_sequence
                    nb_tokens += next_left_tokens
                    j += 1
                else:
                   break
            counter += 1

        inputs = self.tokenizer(new_sequence, return_tensors='pt', padding='max_length',
                                max_length=max_length, truncation=True, add_special_tokens=True, 
                                return_attention_mask=True)

        return {'input_ids': inputs.input_ids.squeeze(0),
                'attention_mask': inputs.attention_mask.squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long),
                'citation_text':citation_text
                }
      
def list_idx(label_mapping, label_list):
   return [label_mapping[label.lower()] for label in label_list]

class CITSEGEncoder(nn.Module):
      def __init__(self, num_heads=3, attention_dim=250):
          super().__init__()
          hidden_size = num_heads * attention_dim 
          self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=3, batch_first=True)

      def forward(self, citseg_embedding):
        citseg_embedding = citseg_embedding.unsqueeze(1)
        encoded = self.encoder_layer(citseg_embedding)
        return encoded.squeeze(1)
      
def main(model_short_name, window, data_repartition, SEED):
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


  if torch.cuda.is_available():
      device = torch.device("cuda") 
  else:
      device = torch.device("cpu")

  citation_sequence_x_jiang, citation_sequence_y_jiang, jiang_citances, jiang_left_contexts, jiang_right_contexts, Jiang_ctx_ind = get_data_list(window)


  models = {'BioBERT': 'dmis-lab/biobert-v1.1', 'SciBERT': 'allenai/scibert_scivocab_uncased', 'RoBERTa-large': 'all-roberta-large-v1', 'RoBERTa' : 'roberta-base'}
  
  model_name = models[model_short_name]
  model = AutoModel.from_pretrained(model_name)
  print(f'Model {model} loaded to device')
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  tokenizer.add_tokens(['CITSEG'], special_tokens=True)
  model.resize_token_embeddings(len(tokenizer))
  
  num_of_epochs = 20
  batch_size = 16
  in_features = model.config.hidden_size
  

  all_labels = ['similar', 'neutral', 'usage', 'cocores', 'motivation', 'basis', 'weakness', 'future', 'support', 'cocogm', 'cocoxy']
  label_mapping = {label: idx for idx, label in enumerate(all_labels)}

  data_to_shuffle = list(zip(citation_sequence_x_jiang, citation_sequence_y_jiang, jiang_citances, jiang_left_contexts, jiang_right_contexts))
  random.shuffle(data_to_shuffle)
  citation_sequence_x_jiang_shuffled, citation_sequence_y_jiang_shuffled, jiang_citances_shuffled, jiang_left_contexts_shuffled, jiang_right_contexts_shuffled= zip(*data_to_shuffle)
  citation_sequence_x_jiang_shuffled = list(citation_sequence_x_jiang_shuffled)
  citation_sequence_y_jiang_shuffled = list(citation_sequence_y_jiang_shuffled)
  jiang_citances_shuffled, jiang_left_contexts_shuffled, jiang_right_contexts_shuffled = list(jiang_citances_shuffled), list(jiang_left_contexts_shuffled), list(jiang_right_contexts_shuffled)

    
  if data_repartition == 'Jiang_all':
    train_ratio = 0.65
    val_ratio = 0.15
    test_ratio = 0.20
      
    total_samples = len(citation_sequence_x_jiang_shuffled)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    x_train, y_train = citation_sequence_x_jiang_shuffled[:train_size], citation_sequence_y_jiang_shuffled[:train_size]
    citance_train, left_train, right_train = jiang_citances_shuffled[:train_size], jiang_left_contexts_shuffled[:train_size], jiang_right_contexts_shuffled[:train_size]

    x_val, y_val = citation_sequence_x_jiang_shuffled[train_size:train_size+val_size], citation_sequence_y_jiang_shuffled[train_size:train_size+val_size]
    citance_val, left_val, right_val = jiang_citances_shuffled[:val_size], jiang_left_contexts_shuffled[:val_size], jiang_right_contexts_shuffled[:val_size]

    x_test, y_test = citation_sequence_x_jiang_shuffled[train_size+val_size:], citation_sequence_y_jiang_shuffled[train_size+val_size:]
    citance_test, left_test, right_test = jiang_citances_shuffled[:test_size], jiang_left_contexts_shuffled[:test_size], jiang_right_contexts_shuffled[:test_size]
      
    labels_idx_train, labels_idx_val, labels_idx_test = list_idx(label_mapping, y_train), list_idx(label_mapping, y_val), list_idx(label_mapping, y_test)
    
    ACC_STEP = 1
      
    citseg_encoder = CITSEGEncoder()    
    citation_model = CitationClassifier(
    scibert=model,  
    citseg_encoder=citseg_encoder,  
    tokenizer=tokenizer,  
    hidden_size=in_features, 
    num_class=len(all_labels),
    device=device)
    
    citation_model.to(device)

    learning_rates = {
    "scibert": 5e-5,
    "custom": 5e-4 
    }

    optimizer = AdamW([
        {"params": citation_model.scibert.parameters(), "lr": learning_rates["scibert"]},
        {"params": citation_model.citseg_encoder.parameters(), "lr": learning_rates["custom"]},
        {"params": citation_model.mlp.parameters(), "lr": learning_rates["custom"]}
    ], weight_decay=0.002)

    loss_fn = nn.CrossEntropyLoss()

    train_dataset = CitationDataset(text_citations=x_train, labels_ind=labels_idx_train, tokenizer=tokenizer, citances=citance_train, left_ctxs=left_train, right_ctxs=right_train, ctx_ind=Jiang_ctx_ind)
    val_dataset = CitationDataset(text_citations=x_val, labels_ind=labels_idx_val, tokenizer=tokenizer, citances=citance_val, left_ctxs=left_val, right_ctxs=right_val, ctx_ind=Jiang_ctx_ind)
    test_dataset = CitationDataset(text_citations=x_test, labels_ind=labels_idx_test, tokenizer=tokenizer, citances=citance_test, left_ctxs=left_test, right_ctxs=right_test, ctx_ind=Jiang_ctx_ind)
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    num_training_steps = (len(train_loader) /ACC_STEP)
    total_steps = len(train_loader) * num_of_epochs
    warmup=0.1
    warmup_steps = math.floor(total_steps * warmup)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

      
    tqdm.pandas()
    best_acc, best_f1, best_macrof1 = 0, 0, 0
    best_epoch =0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    val_F_mesures, train_F = [], []
    val_macros, train_macros = [], []

    for i in tqdm(range(num_of_epochs)):          
      train_loss = training_step(train_loader, citation_model, optimizer, loss_fn, device, ACC_STEP, scheduler)
      train_acc, train_f1, train_loss2, train_F1macro = validation_step(train_loader, citation_model, loss_fn, device)
      val_acc, val_f1, val_loss, val_F1macro = validation_step(val_loader, citation_model, loss_fn, device)

      train_losses.append(train_loss)
      train_accuracies.append(train_acc)
      val_accuracies.append(val_acc)
      val_F_mesures.append(val_f1)
      train_F.append(train_f1)
      val_losses.append(val_loss)
      val_macros.append(val_F1macro)
      train_macros.append(train_F1macro)
      

      if val_acc > best_acc:
          best_acc = val_acc    
      if val_f1 > best_f1:
          best_f1 = val_f1
          best_epoch=i

      if val_f1 > best_macrof1:
          best_macrof1 = val_F1macro

    print(f"{model_short_name} Best val Fweighted : {best_f1} for epoch {best_epoch}")
    print(f"{model_short_name} Best val Fmacro : {best_macrof1}")

    citation_model.eval()

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
          
          preds = citation_model(tokens=input_ids, attention_mask=attention_mask)
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

    print(len(true_labels), len(predictions))
    df = pd.DataFrame(csv_output)
    output_csv_path = f"{model_short_name}_predictions_acc_steps_{ACC_STEP}_repartition_{data_repartition}_ctx_{window}.csv"
    df.to_csv(output_csv_path, index=False)
    
    from sklearn.metrics import classification_report
    report = classification_report(true_labels, predictions, output_dict=True)
    print(report)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Choose the model to vectorize citations")
  parser.add_argument(
        "--model",
        type=str,
        default="SciBERT",
        choices=["SciBERT", "BioBERT", "RoBERTa"],
        help="The model to use. Options: 'SciBERT' (default), 'BioBERT', or 'RoBERTa'."
    )
  parser.add_argument(
        "--window_context",
        type=str,
        default=None,
        help="The window for the right and left context. Example of format : '2-3' for 2 sentences on the left and 3 sentences on the right, or None (default)."
    )

  parser.add_argument(
        "--data_repartition",
        type=str,
        default="Jiang_train-PD_test",
        help="Data repartition : 'Jiang_train-PD_test' ; 'Jiang_all' ; 'Jiang_train-PD_test' ; 'all_crossvalidation' ; 'PD_crossvalidation' "
    )
  parser.add_argument(
        "--seed",
        type=int,
        default=5171,
        help="Learning rate"
    )
  
  args = parser.parse_args()
  main(args.model, args.window_context, args.data_repartition, args.seed)