from torch import nn
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

class CitationClassifier(nn.Module):
    """
    Extract CITSEG representation then do a classification
    """
    def __init__(self, linear_size, model, tokenizer, in_features, num_class):
        super(CitationClassifier, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.in_features = in_features
        self.linear1 = nn.Linear(in_features=self.in_features, out_features=linear_size)
        self.batch_norm1 = nn.BatchNorm1d(num_features=linear_size)
        self.dropout1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(in_features=linear_size, out_features=num_class)
        
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        
    def forward(self, tokens, attention_mask):
        bert_output = self.model(input_ids=tokens, attention_mask=attention_mask)

        last_hidden_state = bert_output.last_hidden_state
        batch_size = tokens.shape[0]
        hidden_dim = bert_output[0].shape[-1]
        citseg_id = self.tokenizer.convert_tokens_to_ids("CITSEG")
        citseg_positions = (tokens == citseg_id).nonzero(as_tuple=False)
        CITSEG_embeddings = torch.zeros((batch_size, hidden_dim), device=last_hidden_state.device)


        for batch_index in range(batch_size):
            citseg_pos = citseg_positions[citseg_positions[:, 0] == batch_index]
            
            if len(citseg_pos) > 0:
                seq_position = citseg_pos[0, 1].item()
                CITSEG_embeddings[batch_index] = last_hidden_state[batch_index, seq_position, :]

        x = self.linear1(CITSEG_embeddings)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return x
    

    
def eval_prediction(y_batch_actual, y_batch_predicted, all_labels):
    """Return batches of accuracy, f1 scores and P, R, F per class."""

    y_batch_actual_np = y_batch_actual.cpu().detach().numpy()
    y_batch_predicted_np = torch.argmax(y_batch_predicted, dim=1).cpu().detach().numpy()
    
    acc = accuracy_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np)
    f1 = f1_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np, average='weighted')
    f1_macro = f1_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np, average='macro')

    precision_per_class = precision_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np, average=None, labels=range(len(all_labels)), zero_division=0)
    recall_per_class = recall_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np, average=None, labels=range(len(all_labels)), zero_division=0)
    f1_per_class = f1_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np, average=None, labels=range(len(all_labels)), zero_division=0)

    return acc, f1, f1_macro, precision_per_class, recall_per_class, f1_per_class
    
def training_step(dataloader, model, optimizer, loss_fn, device, ACC_STEP, scheduler):    
    model.train()
    
    epoch_loss = 0

    for i, batch in enumerate(dataloader):        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
    
        outputs = model(tokens=input_ids, attention_mask=attention_mask)
                        
        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        epoch_loss += loss.item()
        loss = loss / ACC_STEP
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (i+1) % ACC_STEP == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        optimizer.step()
        scheduler.step()
    
    return epoch_loss
        

def validation_step(dataloader, model, loss_fn, device, all_labels):        
    model.eval()
    
    size = len(dataloader)
    f1, acc, total_loss, f1_macro_total = 0, 0, 0, 0

    all_precision_per_class = []
    all_recall_per_class = []
    all_f1_per_class = []
    
    
    with torch.no_grad():
        for batch in dataloader:
            X = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['labels'].to(device)
                
            pred = model(tokens=X, attention_mask=attention_mask)

            loss = loss_fn(pred, y)  
            total_loss += loss.item()
            
            acc_batch, f1_batch, f1_macro_batch, precision_per_class, recall_per_class, f1_per_class = eval_prediction(y.float(), pred, all_labels)                        
            acc += acc_batch
            f1 += f1_batch
            f1_macro_total += f1_macro_batch


            all_precision_per_class.append(precision_per_class)
            all_recall_per_class.append(recall_per_class)
            all_f1_per_class.append(f1_per_class)

        acc = acc/size
        f1 = f1/size
        f1_macro_total = f1_macro_total / size 

        max_classes = max(map(len, all_precision_per_class))

        all_precision_per_class = [np.pad(p, (0, max_classes - len(p)), 'constant', constant_values=np.nan) for p in all_precision_per_class]
        all_recall_per_class = [np.pad(r, (0, max_classes - len(r)), 'constant', constant_values=np.nan) for r in all_recall_per_class]
        all_f1_per_class = [np.pad(f, (0, max_classes - len(f)), 'constant', constant_values=np.nan) for f in all_f1_per_class]

        precision_per_class = np.nanmean(all_precision_per_class, axis=0)
        recall_per_class = np.nanmean(all_recall_per_class, axis=0)
        f1_per_class = np.nanmean(all_f1_per_class, axis=0)

                
    return acc, f1, total_loss, f1_macro_total, precision_per_class, recall_per_class, f1_per_class