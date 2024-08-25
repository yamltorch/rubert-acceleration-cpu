import datasets
import evaluate
import transformers
import sys
task_type = "text-classification"

# Update to the desired model and dataset IDs
model_id = "seara/rubert-tiny2-russian-sentiment"
dataset = datasets.load_dataset('json', data_files='train.json', split='train')  # Adjust split if necessary

# Define the label mapping for sentiment classification
label_mapping = {
    "neutral": 0,
    "positive": 1,
    "negative": 2,
}

# Update the dataset to add a numerical label column
def map_labels(example):
    example['labels'] = label_mapping[example['sentiment']]
    return example

# Apply the label mapping
data = dataset.map(map_labels)
# sys.exit()

metric = evaluate.load("accuracy")
evaluator = evaluate.evaluator(task_type)

def evaluate_pipeline(pipeline):
    results = evaluator.compute(
        model_or_pipeline=pipeline,
        data=data,
        metric=metric,
        label_column="labels",
        label_mapping=label_mapping,
    )
    return results

print("*** Original model")
classifier = transformers.pipeline(task_type, model=model_id, tokenizer=model_id)
results = evaluate_pipeline(classifier)
print(results)

print("*** ONNX")

from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.pipelines import pipeline

# Load and save ONNX model
model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model.save_pretrained("./model_onnx")
tokenizer.save_pretrained("./model_onnx")

classifier_onnx = pipeline(task_type, model=model, tokenizer=tokenizer)
results = evaluate_pipeline(classifier_onnx)
print(results)

print("*** ONNX optimizer")

from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

optimizer = ORTOptimizer.from_pretrained(model)
optimizer.optimize(
    OptimizationConfig(
        optimization_level=99,  # Choose the appropriate optimization level
    ),
    save_dir="./model_onnx",
)

model_optimized = ORTModelForSequenceClassification.from_pretrained(
    "./model_onnx", file_name="model_optimized.onnx"
)
classifier_optimized = pipeline(task_type, model=model_optimized, tokenizer=tokenizer)
results = evaluate_pipeline(classifier_optimized)
print(results)

print("*** ONNX quantizer")

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

quantizer = ORTQuantizer.from_pretrained(model)

# Use a more general quantization config suitable for non-Intel hardware
qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=True)  # Adjust as needed for Ryzen

quantizer.quantize(save_dir="./model_onnx", quantization_config=qconfig)

model_quantized = ORTModelForSequenceClassification.from_pretrained(
    "./model_onnx", file_name="model_quantized.onnx"
)
classifier_quantized = pipeline(task_type, model=model_quantized, tokenizer=tokenizer)
results = evaluate_pipeline(classifier_quantized)
print(results)
