import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
import pandas as pd
import numpy as np
from datasets import DatasetDict, Dataset, load_dataset

# generate reward function with base model
def format_chat_prompt(user_input, system_message="You are a helpful assistant."):
    """
    Formats user input into the chat template format with <|im_start|> and <|im_end|> tags.

    Args:
        user_input (str): The input text from the user.

    Returns:
        str: Formatted prompt for the model.
    """
    
    # Format user message
    user_prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n"
    
    # Start assistant's turn
    assistant_prompt = "<|im_start|>assistant\n"
    
    # Combine prompts
    formatted_prompt = user_prompt + assistant_prompt
    
    return formatted_prompt

# dataset preparation


df = pd.read_csv('/home/genisha_admin/Qwen/data/Qwen2_3_prep.csv')
print(df.head())


def assign_responses(row):
    if row['OutputA_preferred'] == 1.0:
        return pd.Series([row['OutputA'], row['OutputB']])
    else:
        return pd.Series([row['OutputB'], row['OutputA']])


df[['chosen_response', 'rejected_response']] = df.apply(assign_responses, axis=1)

# Rename 'Command' to 'prompt'
df = df.rename(columns={'Command': 'prompt'})
df = df[['prompt', 'chosen_response', 'rejected_response']]

#print(df['chosen_response'])

# Apply formatting to match list of dicts
df['prompt'] = df['prompt'].apply(lambda x: [ {"role": "user", "content": x} ])
df['chosen'] = df['chosen_response'].apply(lambda x: [ {"role": "assistant","content": x} ])
df['rejected'] = df['rejected_response'].apply(lambda x: [ { "role": "assistant","content": x} ])

print('_____________')
# Keep only formatted columns
df_final = df[['prompt', 'chosen', 'rejected']]
print(df_final.head())
#print(df_final['chosen'])
df_final.to_csv('/home/genisha_admin/Qwen/data/Qwen2_test.csv')



# Shuffle dataframe
df_shuffled = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
df_shuffled.to_csv('/home/genisha_admin/Qwen/data/Qwen2_3prep_dpo_ready.csv', index=False)


# 90-10 split
train_size = int(0.9 * len(df_shuffled))
df_train = df_shuffled.iloc[:train_size]
df_valid = df_shuffled.iloc[train_size:]

# Convert to Hugging Face Datasets
train_ds = Dataset.from_pandas(df_train)
valid_ds = Dataset.from_pandas(df_valid)

# Combine into a DatasetDict
dataset = DatasetDict({
    'train': train_ds,
    'valid': valid_ds,
})

#__________________________
# load Model and Tokenizer
#__________________________

#model_path = "/home/genisha_admin/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
model_path = "/home/genisha_admin/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
# Example prompt
prompt = format_chat_prompt(dataset['valid']['prompt'][0][0]['content'])
outputs = generator(prompt, max_length=300, truncation=True, num_return_sequences=1, temperature=0.7)


# DPO Training Config
dpo_config = DPOConfig(
    output_dir="../models/dpo_qwen2_lora", 
    logging_steps=25,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_strategy="epoch",
    eval_strategy="epoch",
    eval_steps=1,
)


# Collator & Trainer

trainer = DPOTrainer(
    model=model,     
    args=dpo_config,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
)


trainer.train()

# trainer.save_model("../models/dpo_qwen2_lora/final")
#tokenizer.save_pretrained("../models/dpo_qwen2_lora/final")

# ft_model = trainer.model

# # Set up text generation pipeline
# generator = pipeline("text-generation", model=ft_model, tokenizer=tokenizer)

# # Example prompt
# prompt = format_chat_prompt(dataset['valid']['prompt'][0][0]['content'])

# # Generate output
# outputs = generator(prompt, max_length=100, truncation=True, num_return_sequences=1, temperature=0.7)

# print(outputs[0]['generated_text'])
