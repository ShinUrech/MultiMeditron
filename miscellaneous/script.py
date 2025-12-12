from multiversity.multicare_dataset import MedicalDatasetCreator
import os
import pandas as pd

mdc = MedicalDatasetCreator(directory = 'medical_datasets')
filters = [{'field': 'license', 'string_list': ['CC0', 'CC BY', 'CC BY-SA', 'CC BY-ND', 'CC BY-NC', 'CC BY-NC-SA', 'CC BY-NC-ND']}]
mdc.create_dataset(dataset_name = 'text_reasoning_dataset', filter_list = filters, dataset_type = 'text')

path = "medical_datasets/text_reasoning_dataset/cases.csv"
df = pd.read_csv(path)
df.to_json('cases.jsonl', orient='records', lines=True)
