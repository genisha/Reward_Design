import pandas as pd
import csv

import os
os.environ["PYTORCH_SDP_ATTENTION"] = "never"

from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to your locally downloaded model
# Path for Qwen 2.5-0.5B
#model_path = "/home/genisha_admin/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
# Path for Qwen 2.5-1.5B
#model_path = "/home/genisha_admin/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"    
# Path for Qwen 2.5-3B
model_path = "/home/genisha_admin/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"


# Load tokenizer and model from the local path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Check if CUDA is available and move the model to GPU if possible
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example prompt to feed to the model
prompt = "Drone moves forward and there is one object in front of the drone. What is your reward function? "

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate output
outputs = model.generate(**inputs, max_length=5000, temperature=0.7, do_sample=True)

# Decode and print the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#print(generated_text)

with open('/home/genisha_admin/Qwen/Qwen2_3_Model.csv', 'a', newline='') as f:
    #f.write(generated_text)
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow([generated_text])





#with open('/home/genisha_admin/Qwen/Qwen2_0_5_Model', 'wb') as fd:
#   fd.write(r.raw.read())

#df = pd.DataFrame(generated_text)
#df.to_csv("/home/genisha_admin/Qwen/Qwen2_0_5_Model.csv", index=False)