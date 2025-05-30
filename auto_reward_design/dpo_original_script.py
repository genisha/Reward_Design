
import torch
import os
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import DatasetDict, Dataset, load_dataset
import re


os.environ["WANDB_PROJECT"] = "Run1"  # name my W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint" 


dataset = load_dataset( 'json', data_files={
    'train': 'data/train_0_5_add.json',
    'valid': 'data/valid_0_5.json'
})

#dataset = load_dataset("shawhin/youtube-titles-dpo")

#model_name = "Qwen/Qwen2.5-0.5B-Instruct"
#model_name = "/home/genisha_admin/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
# Path for Qwen 2.5-1.5B
model_name = "/home/genisha_admin/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306" 
# Path for Qwen 2.5-3B
#model_name = "/home/genisha_admin/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # set pad token

#ft_model_name = model_name.split('/')[1].replace("Instruct", "DPO")

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f'GPU: {device}')
model.to(device)


# Extract part: Qwen2.5-0.5B-Instruct
match = re.search(r'models--[^-]+--([^/]+)', model_name)
if match:
    ft_model_name = match.group(1).replace("Instruct", "DPO")
    print(ft_model_name)
# example: Output: Qwen2.5-0.5B-DPO

training_args = DPOConfig(
    output_dir=ft_model_name, 
    logging_steps=25,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_strategy="epoch",
    eval_strategy="epoch",
    eval_steps=1,
    report_to="wandb"
)


print(f'load data and model successfully')
print(f'dataset: {dataset}')

trainer = DPOTrainer(
    model=model, 
    args=training_args, 
    processing_class=tokenizer, 
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid'],
)
trainer.train()
torch.cuda.empty_cache()
