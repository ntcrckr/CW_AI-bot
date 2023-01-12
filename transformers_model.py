import numpy as np
from transformers import DistilBertTokenizerFast
import torch
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    "./transformers_model",
    num_labels=5
)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
# model.to(device)


class Model:
    def __init__(self):
        pass

    def to_check_result(self, test_encoding):
        input_ids = torch.tensor(test_encoding["input_ids"]).to(device)
        attention_mask = torch.tensor(test_encoding["attention_mask"]).to(device)
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
        y = np.argmax(outputs[0].to('cpu').numpy())
        return y

    def get_prediction(self, review_text: str):
        test_encoding1 = tokenizer(review_text, truncation=True, padding=True)
        input_ids = torch.tensor(test_encoding1["input_ids"]).to(device)
        attention_mask = torch.tensor(test_encoding1["attention_mask"]).to(device)
        op = self.to_check_result(test_encoding1)
        return int(op)+1
