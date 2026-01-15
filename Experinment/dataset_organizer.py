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

def save_json_array(data_list, file_path, indent=4):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, 
                    ensure_ascii=False,
                    indent=indent, 
                    separators=(',', ': ') if indent else None)
        print(f"Successfully wrote the {len(jsonl_data)} objects to {file_name}")
    except Exception as e:
        print(f"Write failure: {str(e)}")
def save_json_simple(jsonl_data, file_name, indent=4):
    with open(file_name, 'w', encoding='utf8') as file:
        for obj in jsonl_data:
            if "\n" in str(obj):
                obj = {k: v.replace("\n", "") if isinstance(v, str) else v 
                    for k, v in obj.items()}
            line = json.dumps(obj, ensure_ascii=False)
            file.write(line + "\n")
        print(f"Successfully wrote the {len(jsonl_data)} objects to {file_name}")

def give_id(file_name, output_file, output_file_f):
    jsonl_data = read_jsonl(file_name)
    to_write = []
    i = 0
    for data in jsonl_data:
        cur_data = {}
        cur_data['id'] = i
        cur_data['problem'] = data['problem']
        cur_data['ground_truth_solution'] = data['ground_truth_solution']
        cur_data['Error CoT'] = data['Error CoT']
        cur_data['Error type'] = data['Error type']
        to_write.append(cur_data)
        i += 1
    save_json_array(to_write, output_file_f)
    save_json_simple(to_write, output_file)
    

give_id('./Confounding_error.jsonl', 
        './Confounding_error/Confounding_error.jsonl',
        './Confounding_error/Confounding_error_f.jsonl')

give_id('./Collider_error.jsonl',
        './Collider_error/Collider_error.jsonl',
        './Collider_error/Collider_error_f.jsonl')

give_id('./Measure_error.jsonl', 
        './Measure_error/Measure_error.jsonl', 
        './Measure_error/Measure_error_f.jsonl')

give_id('./Mediation_error.jsonl', 
        './Mediation_error/Mediation_error.jsonl', 
        './Mediation_error/Mediation_error_f.jsonl')
