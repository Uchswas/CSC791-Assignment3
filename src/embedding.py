import torch
import json
from transformers import AutoTokenizer, AutoModel

#base_model
codebert_base_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
codebert_base_model = AutoModel.from_pretrained("microsoft/codebert-base")

#clone detection model fine-tuned from codebert-base
clone_detection_model = AutoModel.from_pretrained("ljcnju/CodeBertForClone-Detection")

def get_base_model_code_embedding(code_snippet):
    inputs = codebert_base_tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = codebert_base_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    embedding_pool = outputs.pooler_output.squeeze()
    return embedding.numpy().tolist(), embedding_pool.numpy().tolist()

def get_clone_detection_code_embedding(code_snippet):
    inputs = codebert_base_tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = clone_detection_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    embedding_pool = outputs.pooler_output.squeeze()
    return embedding.numpy().tolist(), embedding_pool.numpy().tolist()

with open("../data/code_snippet.json", "r") as file:
    data = json.load(file)

embeddings_dict_base = {}
embeddings_dict_base_pool = {}
for item in data:
    code_id = item["id"]
    if "code1" in item:
        embedding_code1 = get_base_model_code_embedding(item["code1"])
        embeddings_dict_base[f"{code_id}_1"] = embedding_code1[0]
        embeddings_dict_base_pool[f"{code_id}_1"] = embedding_code1[1]
    if "code2" in item:
        embedding_code2 = get_base_model_code_embedding(item["code2"])
        embeddings_dict_base[f"{code_id}_2"] = embedding_code2[0]
        embeddings_dict_base_pool[f"{code_id}_2"] = embedding_code1[1]

with open("../results/code_embeddings/code_embeddings_base_model.json", "w") as file:
    json.dump(embeddings_dict_base, file, indent=4)
with open("../results/code_embeddings/code_embeddings_base_model_pool.json", "w") as file:
    json.dump(embeddings_dict_base_pool, file, indent=4)

print("Embeddings for CodeBERT base model have been saved to results/code_embeddings/code_embeddings_base_model.json and results/code_embeddings/code_embeddings_base_model_pool.json ")

embeddings_dict_clone = {}
embeddings_dict_clone_pool = {}
for item in data:
    code_id = item["id"]
    if "code1" in item:
        embedding_code1 = get_clone_detection_code_embedding(item["code1"])
        embeddings_dict_clone[f"{code_id}_1"] = embedding_code1[0]
        embeddings_dict_clone_pool[f"{code_id}_1"] = embedding_code1[1]
    if "code2" in item:
        embedding_code2 = get_clone_detection_code_embedding(item["code2"])
        embeddings_dict_clone[f"{code_id}_2"] = embedding_code2[0]
        embeddings_dict_clone_pool[f"{code_id}_2"] = embedding_code2[1]

with open("../results/code_embeddings/code_embeddings_clone_detection_model.json", "w") as file:
    json.dump(embeddings_dict_clone, file, indent=4)
with open("../results/code_embeddings/code_embeddings_clone_detection_model_pool.json", "w") as file:
    json.dump(embeddings_dict_clone_pool, file, indent=4)

print("Embeddings for Clone Detection model have been saved to results/code_embeddings/code_embeddings_clone_detection_model.json and results/code_embeddings/code_embeddings_clone_detection_model_pool.json")
