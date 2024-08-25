import json
from datasets import Dataset

# Load JSON data
with open("your_data.json", "r", encoding="utf-8") as file:
    raw_data = json.load(file)

# Map sentiment labels to integers
label_mapping = {
    "neutral": 0,
    "positive": 1,
    "negative": 2,
}

# Preprocess the data
preprocessed_data = []
for entry in raw_data:
    preprocessed_entry = {
        "text": entry["text"],
        "label": label_mapping[entry["sentiment"]],
    }
    preprocessed_data.append(preprocessed_entry)

# Convert the preprocessed data into a Hugging Face Dataset object
dataset = Dataset.from_list(preprocessed_data)


test_dataset = dataset["test"]


# Save the preprocessed datasets (optional, for later use)

test_dataset.save_to_disk("./test_dataset")