import json
import pandas as pd
from tqdm import tqdm
import requests
import time
from pathlib import Path
from collections import defaultdict
import random
from utility import get_paths, get_levels, build_id2node, leaf_ids_under, get_new_leaves, choose_k_by_silhouette, create_linkage, plot_tree_interactive, get_effective_leaves, tree_to_linkage, build_message_paper_task_labels, clean_json_string, add_summary
from promptLLM import open_route_ai_models
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree, inconsistent
import matplotlib.pyplot as plt
from sklearn.cluster import Birch, AgglomerativeClustering
import numpy as np
from hierarchy.kmeans_agglomerative import KMeansAgglomerativeClustering
from RecursiveHierarchicalClustering import RecursiveHierarchicalClustering
from TaxonomyTree import binary_to_multiway_taxonomy, convert_binary_to_multiway
from plot_taxonomy import plot_tree_interactive_tax, get_effective_leaves_tax
from datasets import load_dataset
import torch
import os
from evaluate import get_path_granularity, get_level_granularity, get_novelty, get_leaf_novelty, get_node_novelty, get_parent_child_granularity, get_coverage
import math
import pickle
random.seed(42)  # Set seed


OUTPUT_PATH   = "/u/spa-d4/grad/mfe261/Projects/TeacherApproved/output"
DATA_PATH     = "/u/siddique-d1/Moghis/Taxonomy"
MODEL_ID     =  "qwen/qwen3-235b-a22b-2507"# "" # "qwen/qwen3-235b-a22b-2507"
ENCODER =  "Qwen/Qwen3-Embedding-8B" #Qwen/Qwen3-Embedding-4B #"sentence-transformers/all-mpnet-base-v2" #, "intfloat/e5-large-v2"
COL = "review_complaint"
# initial_cluster_size = 1
#run_kmeans_agglomerative_threshold = 20000
initial_cluster_size = 400
run_kmeans_agglomerative_threshold = 40000  # if the number of labels > threshold, run kmeans_agglomerative otherwise run agglomerative clustering directly
unique_labels = True  # Whether to deduplicate labels before clustering
save_path = f"{DATA_PATH}/embeddings_all.pt"
save_path_unique = f"{DATA_PATH}/embeddings_unique.pt"

# 1- Get the embedding of generated summeries
if not unique_labels:
    if os.path.exists(save_path):
        print(f"Loading cached embeddings from {save_path}")
        data = torch.load(save_path, map_location="cpu", weights_only=False)  # or "cuda" if you want
        all_labels = data["labels"]
        X = data["embeddings"]
    else:
        
        df_loaded = pd.read_excel(f"{DATA_PATH}/privacy_summaries_filtered_all.xlsx")
        results = df_loaded.to_dict(orient="records")

        all_labels = []
        for res in results:
            all_labels.extend(res['privacy/security summaries'].split("; "))
            
        df = pd.read_excel(f"{DATA_PATH}/privacy_summaries_3.xlsx")
        results = df.to_dict(orient="records")

        for res in results:
            val = res.get('privacy/security summaries')
            if isinstance(val, str) and val.strip():
                all_labels.extend(res['privacy/security summaries'].split("; "))
        
        
        enc = SentenceTransformer(ENCODER, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
        X = enc.encode(all_labels, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
        X = normalize(X, norm="l2") # L2-normalize (recommended)
        data = {
            "labels": all_labels,          # list[str]
            "embeddings": X,               # np.ndarray (N, d)
            "encoder": ENCODER,            # optional but useful
        }
        torch.save(data, save_path, pickle_protocol=4)
else:
    if os.path.exists(save_path_unique):
        print(f"Loading cached unique embeddings from {save_path_unique}")
        data = torch.load(save_path_unique, map_location="cpu", weights_only=False)  # or "cuda" if you want
        all_labels = data["labels"]
        X = data["embeddings"]
    else:
        df_loaded = pd.read_excel(f"{DATA_PATH}/privacy_summaries_filtered_all.xlsx")
        results = df_loaded.to_dict(orient="records")

        all_labels = []
        for res in results:
            all_labels.extend(res['privacy/security summaries'].split("; "))
            
        df = pd.read_excel(f"{DATA_PATH}/privacy_summaries_3.xlsx")
        results = df.to_dict(orient="records")

        for res in results:
            val = res.get('privacy/security summaries')
            if isinstance(val, str) and val.strip():
                all_labels.extend(res['privacy/security summaries'].split("; "))
        
        
        
        all_labels = list(set(all_labels))
        enc = SentenceTransformer(ENCODER, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
        X = enc.encode(all_labels, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
        X = normalize(X, norm="l2")

        data = {
            "labels": all_labels,          # list[str]
            "embeddings": X,               # np.ndarray (N, d)
            "encoder": ENCODER,            # optional but useful
        }
        torch.save(data, save_path_unique, pickle_protocol=4)


# 2- Run agglomerative clustering

cluster = RecursiveHierarchicalClustering(initial_cluster_size=initial_cluster_size, run_kmeans_agglomerative_threshold=run_kmeans_agglomerative_threshold).fit(X, labels=all_labels)

leaves = get_effective_leaves(cluster.root, min_distance_threshold=10.0)

# approach two (Height-tolerance collapsing):
TaxaTreeRoot, Taxaleaves = convert_binary_to_multiway(cluster.root, leaves, tol=250.0)

TaxaTreeRoot.summarry = "Privacy/Security"
add_summary(TaxaTreeRoot, Taxaleaves, summarizer_model=MODEL_ID)

fig = plot_tree_interactive_tax(
    TaxaTreeRoot, 
    Taxaleaves,
    figsize=(2200, 800),
    max_labels_display=40,
)
fig.show(renderer='browser')
print("Done!")



# Evaluation
paths = get_paths(TaxaTreeRoot)
path_wise_granularity = get_path_granularity(paths)

valid_items = [(item['score'], len(item['path'].split(" -> "))) 
               for item in path_wise_granularity if item['score'] != -1]

weighted_scores = [score * length for score, length in valid_items]
total_weight = sum(length for _, length in valid_items)
avg_granularity = sum(weighted_scores) / total_weight if total_weight else 0
print("Average Granularity Score for Paths:", avg_granularity)

#valid_granularity_scores = [item['score'] for item in path_wise_granularity if item['score'] != -1]
#avg_granularity = (sum(valid_granularity_scores) / len(valid_granularity_scores)) if valid_granularity_scores else 0



levels = get_levels(TaxaTreeRoot)
level_wise_granularity = get_level_granularity(levels)
valid_level_granularity_scores = [item['score'] for item in level_wise_granularity if item['score'] != -1]
avg_granularity = (sum(valid_level_granularity_scores) / len(valid_level_granularity_scores)) if valid_level_granularity_scores else 0
print("Average Granularity Score for Levels:", avg_granularity)


parent_child_res = [
    {'parent': item['parent'], 'child': sibling}
    for item in levels
    for sibling in item['siblings']
]

level_novelty = get_novelty(parent_child_res)
# Filter out unscored (-1) items
alpha = 0.5          # 0 = pure ratio, 1 = pure count, 0.5 = equal balance
min_novel_target = 500  # the novel count at which count_score plateaus at 1.0
valid_novelty_items = [item for item in level_novelty if item['score'] != -1]
novel_count = sum(1 for item in valid_novelty_items if item['score'] == 1)
total_valid = len(valid_novelty_items)

# Balanced metric: geometric mean of novelty ratio and normalized count
novelty_ratio = (novel_count / total_valid) if total_valid > 0 else 0
count_score = min(math.log(novel_count + 1) / math.log(min_novel_target + 1), 1.0) if novel_count > 0 else 0
balanced_novelty_score = (novelty_ratio ** (1 - alpha)) * (count_score ** alpha) if (novelty_ratio > 0 and count_score > 0) else 0

print("Number of Novel Parent-Child Pairs:", sum(1 for item in level_novelty if item['score'] == 1))
print(f"Novelty Ratio (quality) Parent-Child Pairs:{novelty_ratio}  ({novel_count}/{total_valid} valid pairs are novel)")
print(f"Count Score (quantity) Parent-Child Pairs:{count_score}  (target: {min_novel_target} novel pairs)")
print(f"Balanced Novelty Score (a={alpha}) Parent-Child Pairs: {balanced_novelty_score}")





# Leaf names are derived later, but we need them here — compute early
leaf_names_for_novelty = list({leaf.summarry for leaf in Taxaleaves if leaf.summarry})
leaf_novelty_standalone = get_leaf_novelty(leaf_names_for_novelty)

# Filter out unscored (-1) items
alpha = 0.5          # 0 = pure ratio, 1 = pure count, 0.5 = equal balance
min_novel_target = 500  # the novel count at which count_score plateaus at 1.0
valid_novelty_items = [item for item in leaf_novelty_standalone if item['score'] != -1]
novel_count = sum(1 for item in valid_novelty_items if item['score'] == 1)
total_valid = len(valid_novelty_items)

# Balanced metric: geometric mean of novelty ratio and normalized count
novelty_ratio = (novel_count / total_valid) if total_valid > 0 else 0
count_score = min(math.log(novel_count + 1) / math.log(min_novel_target + 1), 1.0) if novel_count > 0 else 0
balanced_novelty_score = (novelty_ratio ** (1 - alpha)) * (count_score ** alpha) if (novelty_ratio > 0 and count_score > 0) else 0

print("Number of Novel Leaves (standalone):", sum(1 for item in leaf_novelty_standalone if item['score'] == 1))
print(f"Novelty Ratio (quality) Leaves (standalone):{novelty_ratio}  ({novel_count}/{total_valid} valid leaves are novel)")
print(f"Count Score (quantity) Leaves (standalone):{count_score}  (target: {min_novel_target} novel leaves)")
print(f"Balanced Novelty Score (a={alpha}) Leaves (standalone): {balanced_novelty_score}")


# Weighted node novelty: all nodes scored, weighted by node.count (number of labels)
all_nodes_name_count = []
stack = list(TaxaTreeRoot.children)  # skip root — it's an artificial aggregate node
while stack:
    node = stack.pop()
    if node.summarry:
        all_nodes_name_count.append((node.summarry, node.count))
    stack.extend(node.children)

node_names_for_novelty = [name for name, _ in all_nodes_name_count]
node_count_map = {name: count for name, count in all_nodes_name_count}

all_node_novelty = get_node_novelty(node_names_for_novelty)

weighted_sum = 0.0
total_weight = 0
for item in all_node_novelty:
    if item['score'] == -1:
        continue
    w = node_count_map.get(item['node'], 0)
    weighted_sum += item['score'] * w
    total_weight += w

weighted_node_novelty = (weighted_sum / total_weight) if total_weight > 0 else 0.0
novel_node_count = sum(1 for item in all_node_novelty if item['score'] == 1)
valid_node_count = sum(1 for item in all_node_novelty if item['score'] != -1)
print(f"Weighted Node Novelty (all nodes, weighted by count): {weighted_node_novelty:.4f}")
print(f"  Novel nodes: {novel_node_count}/{valid_node_count} valid nodes")






pc_granularity = get_parent_child_granularity(parent_child_res)
valid_pc_granularity = [item['score'] for item in pc_granularity if item['score'] != -1]
avg_pc_granularity = sum(valid_pc_granularity) / len(valid_pc_granularity) if valid_pc_granularity else 0
print("Average Granularity Score for Parent-Child Pairs:", avg_pc_granularity)

# Coverage: fraction of pseudo-labels with best cosine similarity to a leaf >= 0.6
leaf_names = [leaf.summarry for leaf in Taxaleaves if leaf.summarry]
enc = SentenceTransformer(ENCODER, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
leaf_embs_for_coverage = enc.encode(leaf_names, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
leaf_embs_for_coverage = normalize(leaf_embs_for_coverage, norm="l2")
coverage_result = get_coverage(X, leaf_embs_for_coverage, threshold=0.6)
print(f"Coverage: {coverage_result['coverage']:.4f} "
      f"({coverage_result['covered']}/{coverage_result['total_labels']} labels "
      f"above threshold {coverage_result['threshold']})")

# I should save the taxonomy tree so next time I can directly load it and do evaluation without needing to run
with open(f"{DATA_PATH}/root_taxon_no_base_v4.pkl", "wb") as f:
    pickle.dump(TaxaTreeRoot, f)
    
fig.write_html(f"{DATA_PATH}/taxonomy_no_base_v4.html")

# load it back
with open(f"{DATA_PATH}/root_taxon_no_base_v4.pkl", "rb") as f: # root_taxon_new_summarizer, root_taxon_mindist_10_tol_20
    TaxaTreeRoot = pickle.load(f)
    
Taxaleaves = get_effective_leaves_tax(TaxaTreeRoot)
