import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Load the model
model = BertForSequenceClassification.from_pretrained('C:\\Users\\Shadab.Kn\\Desktop\\NLP\\fine_tuned_bert_imdb_model')  #full location of model directory
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt', max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model(inputs['input_ids'], token_type_ids=None, attention_mask=inputs['attention_mask'])
    
    predicted_class = torch.argmax(outputs[0]).item()
    labels = ['Negative', 'Positive']  # Assuming 0 is for Negative and 1 is for Positive sentiment
    predicted_sentiment = labels[predicted_class]
    
    return predicted_sentiment

# Streamlit app
def main():
    st.title('Sentiment Analysis with LLM')
    user_input = st.text_input("Enter a movie review:")
    
    if user_input:
        prediction = predict_sentiment(user_input)
        st.write(f"Predicted sentiment: {prediction}")

if __name__ == '__main__':
    main()
