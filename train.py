import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the text data from the CSV file
csv_file_path = 'DATA.csv'  # Make sure this path is correct and the file exists

# Load the CSV data into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Ensure the DataFrame has 'Question' and 'Answer' columns
if 'Question' not in df.columns or 'Answer' not in df.columns:
    raise ValueError("The CSV file must contain 'Question' and 'Answer' columns.")

# Combine 'Question' and 'Answer' columns into a single 'text' column
df['text'] = df['Question'] + " " + df['Answer']

# Split the DataFrame into training and validation sets
train_df, test_df = train_test_split(df, test_size=0.1)

# Convert pandas DataFrame to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Prepare data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Load the model
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=200,
    evaluation_strategy="steps",  # Evaluate at each logging step
    eval_steps=500,  # Evaluation and logging every 500 steps
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,  # Evaluation dataset
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained('distilgpt2-finetuned')
tokenizer.save_pretrained('distilgpt2-finetuned')

# Load the trained model and tokenizer for generating responses
