import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the model and tokenizer
model_name = "gpt2"  # You can use "gpt2-medium" or "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

def generate_response(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1,
                                no_repeat_ngram_size=2, early_stopping=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Streamlit app layout
st.title("Chatbot with GPT-2")
st.write("### Talk to the chatbot! Type 'exit' to stop the conversation.")

# Input field for user messages
user_input = st.text_input("You:", "")

if user_input:
    if user_input.lower() == 'exit':
        st.write("Chatbot: Goodbye!")
    else:
        response = generate_response(user_input)
        st.write("Chatbot:", response)
