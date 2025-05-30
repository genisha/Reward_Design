import json 

with open('/home/genisha_admin/Qwen/data/train_0_5.json', 'r') as f1, open('/home/genisha_admin/Qwen/data/train_0_5_add.json', 'r') as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

combined_data = data1 + data2

with open('/home/genisha_admin/Qwen/train_0_5_add.json', 'w') as out:
    json.dump(combined_data, out, indent=4)