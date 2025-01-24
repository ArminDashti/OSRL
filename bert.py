from transformers import BertModel, BertTokenizer
import torch.nn as nn

def create_bert_encoder(model_name="bert-base-uncased"):
    model = BertModel.from_pretrained(model_name)
    for param in model.parameters():
        param.requires_grad = False
    tokenizer = BertTokenizer.from_pretrained(model_name)
    hidden_dim = model.config.hidden_size
    return model, tokenizer, hidden_dim


class BertEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(BertEncoder, self).__init__()
        self.model, self.tokenizer, self.hidden_dim = create_bert_encoder(model_name)
    def forward(self, text_commands):
        texts = [" ".join(cmd) for cmd in text_commands]
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**encoded)
        return outputs.pooler_output
    

    