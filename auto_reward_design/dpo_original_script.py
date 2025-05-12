from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from datasets import DatasetDict, Dataset, load_dataset

#dataset = load_dataset("test_data")
#dataset = load_dataset('csv', data_files='test_data/Qwen2_test.csv')

 #90-10 Split
#split_dataset = dataset['train'].train_test_split(test_size=0.1)
dataset = load_dataset( 'json', data_files={
    'train': 'data/Qwen0_5_dpo_converted.json',
    'valid': 'data/reward_val.json'
})
#print(dataset

#dataset = load_dataset("shawhin/youtube-titles-dpo")
dataset['train'].to_json('data_YouTube.json')
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # set pad token

ft_model_name = model_name.split('/')[1].replace("Instruct", "DPO")

training_args = DPOConfig(
    output_dir=ft_model_name, 
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

device = torch.device('cuda:0')

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