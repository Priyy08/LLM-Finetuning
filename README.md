### Fine-Tuning DistilGPT-2 for Data Science Question Answering

#### Project Overview

This project demonstrates the process of fine-tuning the DistilGPT-2 model on a dataset containing questions and answers related to data science. The fine-tuned model is then used for interactive question answering.

#### Project Components

1. **Data Preparation**
   - **CSV Loading**: The dataset (`DATA.csv`) is loaded from a CSV file into a Pandas DataFrame.
   - **DataFrame Manipulation**: Columns named 'Question' and 'Answer' are expected in the CSV. They are combined into a single 'text' column for training.
   - **Train-Test Split**: The dataset is split into training and validation sets using `train_test_split` from scikit-learn.

2. **Dataset Conversion**
   - **Hugging Face Dataset**: Pandas DataFrames are converted to Hugging Face Dataset objects (`train_dataset` and `test_dataset`).

3. **Tokenization**
   - **Tokenizer Initialization**: The GPT2Tokenizer from Hugging Face Transformers library is used to tokenize the text data.
   - **Tokenization Function**: Custom tokenization function `tokenize_function` is defined to tokenize the text data, with options for truncation and padding.

4. **Model Preparation**
   - **Model Loading**: The DistilGPT-2 model (`GPT2LMHeadModel`) and tokenizer (`GPT2Tokenizer`) are loaded from the `distilgpt2` pretrained version.
   - **Data Collation**: `DataCollatorForLanguageModeling` is used for preparing batches during training.

5. **Training Configuration**
   - **Training Arguments**: Settings for the training process are configured using `TrainingArguments`, specifying parameters like number of epochs, batch size, logging intervals, and evaluation strategy.

6. **Training Script** (train.py)

    -> Functionality:

    - Loads the dataset (DATA.csv) and prepares it for training.
    - Tokenizes the data using DistilGPT-2 tokenizer.
    - Defines training arguments and initializes the Trainer object.
    - Fine-tunes the DistilGPT-2 model on the dataset.
    - Saves the fine-tuned model and tokenizer at the end of training.

Output:
Trained model and tokenizer saved in the directory (distilgpt2-finetuned).

7. **Model and Tokenizer Saving**
   - **Saving**: After training, the fine-tuned model and tokenizer are saved using `model.save_pretrained()` and `tokenizer.save_pretrained()`.

8. **Chatbot Script** (chat.py)

     -> Functionality:

     - Loads the fine-tuned model and tokenizer from distilgpt2-finetuned.
     - Defines a function to generate responses based on user inputs using the model.
     - Implements an interactive chat interface where users can ask questions and receive answers from the model.
Usage:
Executes an interactive session where users can communicate with the trained model.
#### Files Included

- `DATA.csv`: Contains the question-answer pairs for training.
- `README.md`: This file, providing an overview of the project and instructions.
- `train.py`: Python script implementing the fine-tuning process and saving the Fine Tuned Model.
- `chat.py` : Python script to provide Chatbot functionality to interact with Fine Tuned Model
#### Usage

**To run the project**:

1. Ensure Python environment with necessary libraries (`transformers`, `datasets`, `pandas`, `scikit-learn`) is set up.
2. Place `DATA.csv` in the correct directory (`csv_file_path` in the script).
3. Execute `train.py` to perform fine-tuning.
4. The model will be automatically trianed and be saved in current working directory as "distilgpt2-finetuned" which contains our trained model and tokenizer.
4. After training, execute `chat.py` again or use `generate_response` and `chat_with_model()` functions to interact with the trained model.

#### Conclusion

This project demonstrates how to leverage transfer learning via fine-tuning with the DistilGPT-2 model for specific domain tasks, such as question answering in the field of data science. Fine-tuning allows customization of pre-trained language models to better suit specific use cases, showcasing the power of large language models in natural language understanding and generation tasks.
