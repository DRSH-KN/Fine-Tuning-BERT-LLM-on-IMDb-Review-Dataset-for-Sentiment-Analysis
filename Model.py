import torch
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.model_selection import train_test_split

# Load IMDb dataset
imdb_data = pd.read_csv('IMDB_Dataset.csv')
#As this is an job assessment, i have reduced the dataset to only 20,000.
imdb_data = imdb_data[:500]

# Use a subset of the data
imdb_data = imdb_data.sample(frac=0.1, random_state=42)

# Prepare data for fine-tuning
texts = imdb_data['review']
labels = imdb_data['sentiment']  # Assuming 'sentiment' column contains labels (0 for negative, 1 for positive)

# Use BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize texts
tokenized_texts = tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt')

# Convert labels to PyTorch tensor
labels = labels.apply(lambda x: 1 if x == 'positive' else 0)  # Convert labels to numeric values
labels = torch.tensor(labels.values)

# Create TensorDataset
dataset = TensorDataset(tokenized_texts.input_ids, tokenized_texts.attention_mask, labels)

# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # 2 labels: positive, negative

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 2)

# Fine-tuning BERT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

model.to(device)

epochs = 2  # Number of epochs for fine-tuning

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids, attention_mask, label = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=label)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)

    # Validation
    model.eval()
    val_loss = 0
    for batch in val_dataloader:
        input_ids, attention_mask, label = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=label)
            loss = outputs.loss
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)

    print(f'Epoch {epoch + 1}/{epochs} - Average Training Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f}')

# Save the fine-tuned model
model.save_pretrained('fine_tuned_bert_imdb_model')
