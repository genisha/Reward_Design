import torch
import os
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re

# Setup W&B environment
os.environ["WANDB_PROJECT"] = "Run2"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# Load dataset
dataset = load_dataset('json', data_files={
    'train': 'data/train_0_5_add.json',
    'valid': 'data/valid_0_5.json'
})

# Load model
model_name = "/home/genisha_admin/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Print model layer names (optional)
print("\n--- Model Layers ---")
for name, _ in model.named_parameters():
    print(name)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze final transformer block and lm_head (adjust block index if needed)
for name, param in model.named_parameters():
    if "model.layers.23" in name or "lm_head" in name:  # Adjust block index if model has fewer layers
        param.requires_grad = True

# Print CUDA availability
print("\n--- CUDA Info ---")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Extract output dir name
match = re.search(r'models--[^-]+--([^/]+)', model_name)
if match:
    ft_model_name = match.group(1).replace("Instruct", "DPO")
else:
    ft_model_name = "DPO-Output"

# Training configuration
training_args = DPOConfig(
    output_dir=ft_model_name,
    logging_dir="./logs",  # <--- TensorBoard logs go here
    logging_steps=10,      # log more frequently for better graphs
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_strategy="epoch",
    eval_strategy="epoch",
    eval_steps=1,
    report_to=["tensorboard"],  # <--- enable TensorBoard reporting
)



print('\n--- Dataset Loaded ---')
print(dataset)

# Initialize trainer
trainer = DPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid'],
)

# Train
try:
    trainer.train()
    print(f'Traning completed')
except Exception as e:
    print(f'TRaning failed')
    print(e)
# Cleanup
torch.cuda.empty_cache()

print(f"Model and checkpoints saved to: {training_args.output_dir}")

from datetime import datetime
print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
