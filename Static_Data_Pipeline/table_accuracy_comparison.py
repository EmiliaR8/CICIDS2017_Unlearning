# import json
# import os
# import re
# import pandas as pd
# from pathlib import Path

# # ---------------------------------------------------------
# # Configuration
# # ---------------------------------------------------------

# JSON_FILES = [
#     "./results/MADAR/0723_330_MADAR_Task0Clean_True_mixed.json",
#     "es_work_smoketest.json",
#     "./results/MADAR+Unlearn/0724_930_MadarUNLEARN_Task0Clean_True_mixed_seed42.json"
# ]

# OUTPUT_CSV = "per_task_accuracy_comparison.csv"

# # ---------------------------------------------------------
# # Helper Functions
# # ---------------------------------------------------------

# def is_number(x):
#     return isinstance(x, (int, float))


# def looks_like_task_key(key):
#     """
#     Matches:
#         task_0
#         task0
#         Task 0
#         task-1
#         0
#         12
#     """
#     if isinstance(key, int):
#         return True

#     key = str(key).strip()

#     if re.fullmatch(r"\d+", key):
#         return True

#     if re.fullmatch(r"task[\s_-]*\d+", key, flags=re.IGNORECASE):
#         return True

#     return False


# def normalize_task_name(key):
#     key = str(key)

#     m = re.search(r"(\d+)", key)
#     if m:
#         return f"Task {int(m.group(1))}"

#     return key


# # ---------------------------------------------------------
# # Recursive search
# # ---------------------------------------------------------

# def find_task_accuracy(obj, path="root"):
#     """
#     Returns every dictionary that looks like

#     {
#         task_0: 0.95,
#         task_1: 0.92,
#         ...
#     }

#     regardless of where it is located.
#     """

#     matches = []

#     if isinstance(obj, dict):

#         numeric_task_entries = {
#             normalize_task_name(k): v
#             for k, v in obj.items()
#             if looks_like_task_key(k) and is_number(v)
#         }

#         if len(numeric_task_entries) >= 2:
#             matches.append((path, numeric_task_entries))

#         for k, v in obj.items():
#             matches.extend(find_task_accuracy(v, f"{path}.{k}"))

#     elif isinstance(obj, list):

#         for i, item in enumerate(obj):
#             matches.extend(find_task_accuracy(item, f"{path}[{i}]"))

#     return matches


# # ---------------------------------------------------------
# # Read files
# # ---------------------------------------------------------

# rows = []

# for file in JSON_FILES:

#     with open(file, "r") as f:
#         data = json.load(f)

#     matches = find_task_accuracy(data)

#     if len(matches) == 0:
#         print(f"No task accuracy found in {file}")
#         continue

#     # Use the largest task dictionary found
#     path, task_dict = max(matches, key=lambda x: len(x[1]))

#     row = {
#         "File": Path(file).stem,
#         "Found At": path
#     }

#     row.update(task_dict)

#     rows.append(row)

# # ---------------------------------------------------------
# # Build comparison table
# # ---------------------------------------------------------

# df = pd.DataFrame(rows)

# task_columns = sorted(
#     [c for c in df.columns if c.startswith("Task")],
#     key=lambda x: int(re.search(r"\d+", x).group())
# )

# df = df[["File", "Found At"] + task_columns]

# print(df)

# df.to_csv(OUTPUT_CSV, index=False)

# print(f"\nSaved to {OUTPUT_CSV}")



import pickle
import torch  # Uncomment if the pickle contains PyTorch tensors
import pandas as pd

file_path = './Contrastive_Drift/red_agent_output/red_agent_perturbed_dataset.pkl'

# with open(file_path, 'rb') as f:
#     data = pickle.load(f)

#     import pandas as pd

data = pd.read_pickle(file_path)

print("="*40)
print(f"Data Type: {type(data)}")
print("="*40)

if isinstance(data, dict):
    print(f"Dictionary Keys: {list(data.keys())}")
    for key, val in data.items():
        if hasattr(val, 'shape'):
            print(f"  -> Key '{key}': Shape {val.shape}, Type {type(val)}")
        elif hasattr(val, '__len__') and not isinstance(val, (str, bytes)):
            print(f"  -> Key '{key}': Length {len(val)}, Type {type(val)}")
        else:
            print(f"  -> Key '{key}': Value {val} ({type(val)})")

elif isinstance(data, pd.DataFrame):
    print("\nDataFrame Info:")
    print(data.info())
    print("\nFirst 5 Rows:")
    print(data.head())

else:
    print("\nSample Content:")
    print(data)