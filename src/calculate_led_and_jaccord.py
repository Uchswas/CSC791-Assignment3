import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from itertools import combinations
import pandas as pd
from Levenshtein import distance as levenshtein_distance  

def jaccard_similarity(str1, str2):
    vectorizer = CountVectorizer(binary=True).fit([str1, str2])
    vec1, vec2 = vectorizer.transform([str1, str2]).toarray()
    return jaccard_score(vec1, vec2)

with open('../data/code_snippet.json', 'r') as file:
    data = json.load(file)

all_snippets = []
for entry in data:
    id_ = entry['id']
    all_snippets.append((f"{id_}_1", entry['code1']))
    all_snippets.append((f"{id_}_2", entry['code2']))

results = []

for (label1, code1), (label2, code2) in combinations(all_snippets, 2):
    lev_dist = levenshtein_distance(code1, code2)
    jac_sim = jaccard_similarity(code1, code2)
    
    results.append({
        "Pair": f"({label1}, {label2})",
        "Levenshtein Distance": lev_dist,
        "Jaccard Similarity": jac_sim
    })

results_df = pd.DataFrame(results)
results_df.to_excel('../results/jaccord_and_led_distance.xlsx', index=False)
