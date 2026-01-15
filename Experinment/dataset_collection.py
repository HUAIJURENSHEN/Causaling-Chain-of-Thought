import pandas as pd
import json
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            stripped_line = line.strip()
            obj = json.loads(stripped_line)
            data.append(obj)
    return data

def save_json_simple(jsonl_data, file_name, indent=4):
    with open(file_name, 'w', encoding='utf8') as file:
        for obj in jsonl_data:
            if "\n" in str(obj):
                obj = {k: v.replace("\n", "") if isinstance(v, str) else v 
                    for k, v in obj.items()}
            line = json.dumps(obj, ensure_ascii=False)
            file.write(line + "\n")
        print(f"Successfully wrote the {len(jsonl_data)} objects to {file_name}")

output_file = './Experiment/Data_collection.jsonl'

to_write = []
jsonl_data = read_jsonl('./Collider_error/Collider_error.jsonl')
for i in range(300):
    to_write.append(jsonl_data[i])

jsonl_data = read_jsonl('./Confounding_error/Confounding_error.jsonl')
for i in range(300):
    to_write.append(jsonl_data[i])

jsonl_data = read_jsonl('./Measure_error/Measure_error.jsonl')
for i in range(300):
    to_write.append(jsonl_data[i])

jsonl_data = read_jsonl('./Mediation_error/Mediation_error.jsonl')
for i in range(300):
    to_write.append(jsonl_data[i])

i = 0
for ds in to_write:
    ds['id'] = i
    i += 1

save_json_simple(to_write, output_file)