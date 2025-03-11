from torch import nn
import torch
from sklearn.metrics import f1_score, accuracy_score




'''class CitationClassifier(nn.Module):
    #OPTION 2 : reduction apres le pooling
    def __init__(self, hidden_size, citseg_encoder, scibert, tokenizer, num_class, device):
        super(CitationClassifier, self).__init__()
        self.scibert = scibert
        self.tokenizer = tokenizer
        self.citseg_encoder = citseg_encoder
        self.num_classes = num_class
        self.dim_reduction_layer = nn.Linear(768, 250)

        self.bert_output_size = 768
        self.feature_size = 250 * 2  # Concaténation de CITSEG + contexte
        self.linear_size = self.feature_size * 2

        self.mlp = nn.Sequential(
            nn.Linear(self.feature_size, self.linear_size),
            nn.ReLU(),
            nn.Linear(self.linear_size, self.linear_size),
            nn.ReLU(),
            nn.Linear(self.linear_size, num_class) 
        )

    def ctx_pooler(self, tokens, hidden_states, tokenizer):
        #on prend les embeddings des tokens avant et après @ qui a été utilisé ici pour séparer la citance des contextes
        batch_size, seq_len, hidden_dim = hidden_states.shape
        at_token_id = tokenizer.convert_tokens_to_ids("@")
        at_positions = (tokens == at_token_id).nonzero(as_tuple=False)
        pooled_ctx_embeddings = []
        for batch_idx in range(batch_size):
            at_pos = at_positions[at_positions[:, 0] == batch_idx][:, 1]  
            start_idx, end_idx = at_pos.tolist()
            pooled_embedding = []

            for token_idx in range(seq_len):
                token_id = tokens[batch_idx, token_idx].item()
                token_str = tokenizer.convert_ids_to_tokens(token_id)
                embedding = hidden_states[batch_idx, token_idx]

                if token_idx < start_idx or token_idx > end_idx:
                    pooled_embedding.append(embedding)

            stacked_tensor = torch.stack(pooled_embedding)
            max_pooled, indices = stacked_tensor.max(dim=0)
            pooled_ctx_embeddings.append(max_pooled)

        return torch.stack(pooled_ctx_embeddings)

    def forward(self, tokens, attention_mask):
        bert_output = self.scibert(input_ids=tokens, attention_mask=attention_mask)
        last_hidden_state = bert_output.last_hidden_state

        pooled_ctx = self.ctx_pooler(tokens=tokens, hidden_states=last_hidden_state, tokenizer=self.tokenizer)
        citseg_id = self.tokenizer.convert_tokens_to_ids("CITSEG")
        citseg_positions = (tokens == citseg_id).nonzero(as_tuple=False)

        batch_size = tokens.shape[0]
        CITSEG_embeddings = torch.zeros((batch_size, 768), device=last_hidden_state.device)

        for batch_index in range(batch_size):
            citseg_pos = citseg_positions[citseg_positions[:, 0] == batch_index]
            if len(citseg_pos) > 0:
                seq_position = citseg_pos[0, 1].item()
                CITSEG_embeddings[batch_index] = last_hidden_state[batch_index, seq_position, :]

        pooled_ctx = self.dim_reduction_layer(pooled_ctx)
        CITSEG_embeddings = self.dim_reduction_layer(CITSEG_embeddings)

        feature_vec = torch.cat([pooled_ctx, CITSEG_embeddings], dim=1)
        

        return self.mlp(feature_vec)'''



'''class CitationClassifier(nn.Module):
    ##OPTION 1 : reduction avant pooling
    def __init__(self, hidden_size, citseg_encoder, scibert, tokenizer, num_class, device):
        super(CitationClassifier, self).__init__()
        self.scibert = scibert
        self.tokenizer = tokenizer
        self.citseg_encoder = citseg_encoder
        self.num_classes = num_class
        self.dim_reduction_layer = nn.Linear(768, 250)

        self.bert_output_size = 250
        self.feature_size = self.bert_output_size * 2  
        self.linear_size = self.bert_output_size * 4

        self.mlp = nn.Sequential(
            nn.Linear(self.feature_size, self.linear_size),
            nn.ReLU(),
            nn.Linear(self.linear_size, self.linear_size),
            nn.ReLU(),
            nn.Linear(self.linear_size, num_class) 
        )

    def ctx_pooler(self, tokens, hidden_states, tokenizer):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        at_token_id = tokenizer.convert_tokens_to_ids("@")
        at_positions = (tokens == at_token_id).nonzero(as_tuple=False)
        pooled_ctx_embeddings = []


        for batch_idx in range(batch_size):
            at_pos = at_positions[at_positions[:, 0] == batch_idx][:, 1]

            start_idx, end_idx = at_pos.tolist()
            pooled_embedding = []

            for token_idx in range(seq_len):
                token_id = tokens[batch_idx, token_idx].item()
                token_str = tokenizer.convert_ids_to_tokens(token_id)
                embedding = hidden_states[batch_idx, token_idx]

                if token_idx < start_idx or token_idx > end_idx:
                    pooled_embedding.append(embedding)

            stacked_tensor = torch.stack(pooled_embedding)
            max_pooled, indices = stacked_tensor.max(dim=0)
            pooled_ctx_embeddings.append(max_pooled)


        return torch.stack(pooled_ctx_embeddings)

    def forward(self, tokens, attention_mask):
        bert_output = self.scibert(input_ids=tokens, attention_mask=attention_mask)
        last_hidden_state = self.dim_reduction_layer(bert_output.last_hidden_state)

        pooled_ctx = self.ctx_pooler(tokens=tokens, hidden_states=last_hidden_state, tokenizer=self.tokenizer)
        citseg_id = self.tokenizer.convert_tokens_to_ids("CITSEG")
        citseg_positions = (tokens == citseg_id).nonzero(as_tuple=False)

        batch_size = tokens.shape[0]
        CITSEG_embeddings = torch.zeros((batch_size, 250), device=last_hidden_state.device)

        for batch_index in range(batch_size):
            citseg_pos = citseg_positions[citseg_positions[:, 0] == batch_index]
            if len(citseg_pos) > 0:
                seq_position = citseg_pos[0, 1].item()
                CITSEG_embeddings[batch_index] = last_hidden_state[batch_index, seq_position, :]

        feature_vec = torch.cat([pooled_ctx, CITSEG_embeddings], dim=1)

        return self.mlp(feature_vec)'''
    

class CitationClassifier(nn.Module):
    #option 3 : with transformer layer sur citseg
    def __init__(self, scibert, citseg_encoder, tokenizer, device, num_class, hidden_size=768, citseg_size=750):
        super().__init__()
        self.scibert = scibert
        self.citseg_encoder = citseg_encoder
        self.tokenizer = tokenizer  
        self.device = device
        
        
        input_dim = 1518
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, num_class)
        )

    def ctx_pooler(self, tokens, hidden_states):       
        batch_size, seq_len, hidden_dim = hidden_states.shape

        at_token_id = self.tokenizer.convert_tokens_to_ids("@")
        at_positions = (tokens == at_token_id).nonzero(as_tuple=False)

        pooled_ctx_embeddings = []

        for batch_idx in range(batch_size):
            at_pos = at_positions[at_positions[:, 0] == batch_idx][:, 1]

            if len(at_pos) >= 2:
                start_idx, end_idx = at_pos.tolist()
            else:
                start_idx, end_idx = 0, seq_len

            pooled_embedding = []
            for token_idx in range(seq_len):
                if token_idx < start_idx or token_idx > end_idx:
                    pooled_embedding.append(hidden_states[batch_idx, token_idx])

            if pooled_embedding:
                stacked_tensor = torch.stack(pooled_embedding)
                max_pooled, _ = stacked_tensor.max(dim=0)
            else:
                max_pooled = torch.zeros(hidden_dim, device=hidden_states.device)

            pooled_ctx_embeddings.append(max_pooled)

        return torch.stack(pooled_ctx_embeddings)

    def forward(self, tokens, attention_mask):
        outputs = self.scibert(tokens, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        pooled_context = self.ctx_pooler(tokens, hidden_states)
        citseg_id = self.tokenizer.convert_tokens_to_ids("CITSEG")
        citseg_positions = (tokens == citseg_id).nonzero(as_tuple=False)
        self.proj_layer = nn.Linear(768, 750).to(self.device)

        batch_size = tokens.shape[0]
        CITSEG_embeddings = torch.zeros((batch_size, 750), device=hidden_states.device) 

        for batch_index in range(batch_size):
            citseg_pos = citseg_positions[citseg_positions[:, 0] == batch_index]
            
            if len(citseg_pos) > 0:
                seq_position = citseg_pos[0, 1].item()
                CITSEG_embeddings[batch_index] = self.proj_layer(hidden_states[batch_index, seq_position, :])

        citseg_encoded = self.citseg_encoder(CITSEG_embeddings)

        combined_features = torch.cat([pooled_context, citseg_encoded], dim=1)

        return self.mlp(combined_features)
    
def eval_prediction(y_batch_actual, y_batch_predicted):
    """retourne batch d accuracy et de F1 weighted et macro"""
    y_batch_actual_np = y_batch_actual.cpu().detach().numpy()
    y_batch_predicted_np = torch.argmax(y_batch_predicted, dim=1).cpu().detach().numpy()  
    
    acc = accuracy_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np)
    f1 = f1_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np, average='weighted')
    f1_macro = f1_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np, average='macro')

    return acc, f1, f1_macro
    
def training_step(dataloader, model, optimizer, loss_fn, device, ACC_STEP, scheduler):
    model.train()

    epoch_loss = 0
    size = len(dataloader.dataset)

    for i, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(tokens=input_ids, attention_mask=attention_mask)

        loss = loss_fn(outputs, labels)
        epoch_loss += loss.item()
        loss = loss / ACC_STEP
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (i+1) % ACC_STEP == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    return epoch_loss


def validation_step(dataloader, model, loss_fn, device):

    model.eval()

    size = len(dataloader)
    f1, acc, total_loss, f1_macro_total = 0, 0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            X = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['labels'].to(device)

            pred = model(tokens=X, attention_mask=attention_mask)

            loss = loss_fn(pred, y)
            total_loss += loss.item()

            acc_batch, f1_batch, f1_macro_batch = eval_prediction(y.float(), pred)
            acc += acc_batch
            f1 += f1_batch
            f1_macro_total += f1_macro_batch


        acc = acc/size
        f1 = f1/size
        f1_macro_total = f1_macro_total / size
        print("val loss : ", total_loss)

    return acc, f1, total_loss, f1_macro_total