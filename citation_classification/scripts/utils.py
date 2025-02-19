import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def list_idx(label_mapping, label_list):
   return [label_mapping[label.lower()] for label in label_list]

 


def plot_metric_evolution(metric_name, train_values, val_values, num_epochs, model_short_name, learning_rate, ACC_STEP, window):
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_epochs), train_values, label=f'Train {metric_name}', color='blue')
    plt.plot(range(num_epochs), val_values, label=f'Validation {metric_name}', color='red')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Evolution')
    plt.legend()
    plt.tight_layout()
    
    filename = f"{metric_name}_{model_short_name}_plot_{learning_rate}_accseteps{ACC_STEP}_ctx_{window}.png"
    plt.savefig(filename)
    plt.close()


class CitationDataset(Dataset):
    def __init__(self, text_citations : list, labels_ind : list, tokenizer, device):
        self.text_citations = text_citations
        self.labels_ind = labels_ind
        self.max_token_len = 512
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.labels_ind)

    def __getitem__(self, idx):
        citation_text = self.text_citations[idx]
        label = self.labels_ind[idx]
        inputs = self.tokenizer(citation_text, return_tensors='pt', padding='max_length', max_length=self.max_token_len, truncation=True, add_special_tokens=True, return_attention_mask=True)

        return {'input_ids': inputs.input_ids.squeeze(0), 'attention_mask': inputs.attention_mask.squeeze(0), 'labels': torch.tensor(label, dtype=torch.long), 'citation_text':citation_text}

