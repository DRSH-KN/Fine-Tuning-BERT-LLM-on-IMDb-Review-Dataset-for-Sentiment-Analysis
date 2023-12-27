import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Load the fine-tuned BERT model 
model_path = 'fine_tuned_bert_imdb_model' 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)

# Load and preprocess the validation dataset - using the same dataset, but different part for validation. i have chosed only 500 entries
validation_data = pd.read_csv('IMDB_Dataset.csv')
validation_data = validation_data[40000:40500]#just 500 entries 

texts = validation_data['review']
labels = validation_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
labels = torch.tensor(labels.values)

# Tokenize and create TensorDataset
tokenized_texts = tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt')
dataset = TensorDataset(tokenized_texts['input_ids'], tokenized_texts['attention_mask'], labels)
val_dataloader = DataLoader(dataset, batch_size=8)

# Evaluation function
def evaluate_model(model, val_dataloader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, label = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predicted_class = torch.argmax(outputs.logits, dim=1)

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted_class.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return accuracy, precision, recall, f1

# Evaluate the loaded model
accuracy, precision, recall, f1 = evaluate_model(model, val_dataloader)

# Print performance metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
