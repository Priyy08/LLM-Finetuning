from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

model_path = 'distilgpt2-finetuned'
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

def generate_response(prompt, model, tokenizer, max_length=100, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    responses = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]
    return responses

def chat_with_model():
    print("Welcome to the chat! Type 'exit' to end the conversation.")
    while True:
        prompt = input("You: ")
        if prompt.lower() == 'exit':
            print("Goodbye!")
            break
        responses = generate_response(prompt, model, tokenizer)
        for i, response in enumerate(responses):
            print(f"Model: {response}")

# Start the chat
chat_with_model()