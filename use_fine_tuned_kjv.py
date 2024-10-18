from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-kjv-model")
model = AutoModelForCausalLM.from_pretrained("./fine-tuned-kjv-model")

# Input prompt for generation
input_text = "In the beginning"
inputs = tokenizer(input_text, return_tensors="pt", padding=True)

# Generate text with adjusted parameters
outputs = model.generate(
    inputs['input_ids'],
    max_length=300,               # Adjust the output length as needed
    num_return_sequences=1,        # Number of sequences to return
    no_repeat_ngram_size=2,        # Avoid repeating n-grams
    do_sample=True,                # Use sampling instead of greedy decoding
    top_k=50,                      # Limits the tokens to the top K most likely
    top_p=0.90,                    # Nucleus sampling (top-p sampling)
    temperature=0.7                # Adjust to control randomness (lower = less random)
)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

