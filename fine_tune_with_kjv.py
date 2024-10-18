from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Read the text file
with open("kjv.txt", "r") as f:
    text = f.read()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding

# Tokenize the text (split into smaller sequences, longer text)
tokenized_text = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

# Debug: Check the tokenized text
print(f"Tokenized text shape: {tokenized_text['input_ids'].shape}")

# Define a custom dataset class
class BibleDataset(Dataset):
    def __init__(self, tokenized_text, block_size=512):
        self.input_ids = tokenized_text['input_ids']
        self.block_size = block_size
        self.examples = []

        # Splitting text into examples by block_size
        for i in range(0, self.input_ids.size(1) - block_size + 1, block_size):
            self.examples.append({
                'input_ids': self.input_ids[:, i:i+block_size].squeeze(0),
                'labels': self.input_ids[:, i:i+block_size].squeeze(0),
            })

        # Debug: Check the number of examples
        print(f"Number of examples: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

# Ensure tokenized text has at least one token
if tokenized_text['input_ids'].shape[1] == 0:
    raise ValueError("The tokenized text is empty. Check your input text and tokenization process.")

# Instantiate the dataset
bible_dataset = BibleDataset(tokenized_text)

# Debug: Check if dataset is populated
if len(bible_dataset) == 0:
    raise ValueError("The dataset is empty. Ensure the text is tokenized correctly and that there is data to train on.")

# Load the pre-trained model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=30,
    per_device_train_batch_size=4,  # Adjust based on memory availability
    save_steps=1000,
    save_total_limit=2,
    logging_dir='./logs',  # Directory for logs
    logging_steps=10,
)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=bible_dataset,
    tokenizer=tokenizer,  # Include the tokenizer to ensure the labels are handled correctly
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-kjv-model")
tokenizer.save_pretrained("./fine-tuned-kjv-model")  # Save the tokenizer as well

