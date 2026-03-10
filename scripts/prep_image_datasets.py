from datasets import load_from_disk
import os

LOCAL_DATASETS = {
    "eye_dataset":  "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/eye_dataset",
    "skin_dataset": "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/skin_dataset",
}

for name, path in LOCAL_DATASETS.items():
    print("\n" + "=" * 60)
    print("Dataset : " + name)
    print("Path    : " + path)
    if not os.path.exists(path):
        print("ERROR: path does not exist - skipping.")
        continue
    raw = load_from_disk(path)
    # Handle both plain Dataset and DatasetDict
    from datasets import DatasetDict
    if isinstance(raw, DatasetDict):
        print("Splits  : " + str(list(raw.keys())))
        ds = raw["train"]
    else:
        ds = raw
    print("Rows    : " + str(len(ds)))
    print("Columns : " + str(ds.column_names))
    print("Features: " + str(ds.features))
    if len(ds) > 0:
        sample = ds[0]
        for col, val in sample.items():
            preview = str(val)[:120]
            print("  [" + col + "] type=" + type(val).__name__ + "  preview=" + preview)

print("\nDone.")
