from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Download model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer locally
model.save_pretrained("./Meta-Llama-3-8B-Instruct")
tokenizer.save_pretrained("./Meta-Llama-3-8B-Instruct")
