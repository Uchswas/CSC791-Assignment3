import torch
import json
from transformers import AutoTokenizer, AutoModel

# Initialize models and tokenizer with `attn_implementation="eager"`
codebert_base_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
codebert_base_model = AutoModel.from_pretrained("microsoft/codebert-base", attn_implementation="eager")
clone_detection_model = AutoModel.from_pretrained("ljcnju/CodeBertForClone-Detection", attn_implementation="eager")

# Embedding methods

# 1. Mean Pooling
def get_mean_pooled_embedding(model, tokenizer, code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    mean_pooled_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return mean_pooled_embedding.numpy().tolist()

# 2. Pooler Output
def get_pooler_output_embedding(model, tokenizer, code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        pooler_output_embedding = outputs.pooler_output.squeeze()
        return pooler_output_embedding.numpy().tolist()
    else:
        return None  # Skip if pooler_output is not available

# 3. Attention-Based Pooling (fixed)
def get_attention_pooled_embedding(model, tokenizer, code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    # Average attention weights across heads
    attention_weights = outputs.attentions[-1].mean(dim=1)  # Shape: [batch_size, sequence_length, sequence_length]
    # Sum over sequence length to get weight per token
    attention_weights = attention_weights.mean(dim=-1).unsqueeze(-1)  # Shape: [batch_size, sequence_length, 1]
    # Weighted sum of last hidden state
    weighted_embedding = (outputs.last_hidden_state * attention_weights).sum(dim=1).squeeze()
    return weighted_embedding.numpy().tolist()

# 4. Max Pooling Over Last Hidden States
def get_max_pooled_embedding(model, tokenizer, code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    max_pooled_embedding = outputs.last_hidden_state.max(dim=1).values.squeeze()
    return max_pooled_embedding.numpy().tolist()

# 5. Concatenation of Multiple Layers
def get_concat_layer_embedding(model, tokenizer, code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-4:]  # Use last 4 layers
    concat_embedding = torch.cat([state.mean(dim=1) for state in hidden_states], dim=1).squeeze()
    return concat_embedding.numpy().tolist()

# 6. Intermediate Layer Embedding
def get_intermediate_layer_embedding(model, tokenizer, code_snippet, layer_index=-8):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    intermediate_embedding = outputs.hidden_states[layer_index].mean(dim=1).squeeze()  # Choose any intermediate layer
    return intermediate_embedding.numpy().tolist()

# Processing function to apply all embedding methods to a code snippet
def process_code_snippets(data, model, tokenizer):
    embeddings = {
        "mean_pooled": {},
        "pooler_output": {},
        "attention_pooled": {},
        "max_pooled": {},
        "concat_layers": {},
        "intermediate_layer": {}
    }

    for item in data:
        code_id = item["id"]
        for code_key in ["code1", "code2"]:
            if code_key in item:
                code_snippet = item[code_key]
                
                # Mean Pooled Embedding
                embeddings["mean_pooled"][f"{code_id}_{code_key}"] = get_mean_pooled_embedding(
                    model, tokenizer, code_snippet)
                
                # Pooler Output Embedding (skip if None)
                pooler_embedding = get_pooler_output_embedding(model, tokenizer, code_snippet)
                if pooler_embedding is not None:
                    embeddings["pooler_output"][f"{code_id}_{code_key}"] = pooler_embedding
                
                # Attention Pooled Embedding
                embeddings["attention_pooled"][f"{code_id}_{code_key}"] = get_attention_pooled_embedding(
                    model, tokenizer, code_snippet)
                
                # Max Pooled Embedding
                embeddings["max_pooled"][f"{code_id}_{code_key}"] = get_max_pooled_embedding(
                    model, tokenizer, code_snippet)
                
                # Concatenated Layer Embedding
                embeddings["concat_layers"][f"{code_id}_{code_key}"] = get_concat_layer_embedding(
                    model, tokenizer, code_snippet)
                
                # Intermediate Layer Embedding
                embeddings["intermediate_layer"][f"{code_id}_{code_key}"] = get_intermediate_layer_embedding(
                    model, tokenizer, code_snippet)

    return embeddings

# Load code snippets
with open("../data/code_snippet.json", "r") as file:
    data = json.load(file)

# Generate embeddings for base model
base_model_embeddings = process_code_snippets(data, codebert_base_model, codebert_base_tokenizer)

# Save base model embeddings to JSON
for method, embeddings in base_model_embeddings.items():
    with open(f"../results/code_embeddings/code_embeddings_base_model_{method}.json", "w") as file:
        json.dump(embeddings, file, indent=4)

print("Base model embeddings have been saved to ../results/code_embeddings/")

# Generate embeddings for clone detection model
clone_model_embeddings = process_code_snippets(data, clone_detection_model, codebert_base_tokenizer)

# Save clone detection model embeddings to JSON
for method, embeddings in clone_model_embeddings.items():
    with open(f"../results/code_embeddings/code_embeddings_clone_detection_model_{method}.json", "w") as file:
        json.dump(embeddings, file, indent=4)

print("Clone detection model embeddings have been saved to ../results/code_embeddings/")
