import pathlib
import pandas as pd
from tqdm import tqdm
from rapidfuzz import process, fuzz
from sklearn.metrics import cohen_kappa_score
import ast
from collections import defaultdict

# mark_spans_in_tokens

def mark_extracted_tokens(labels, summary_spans, tokens, threshold=80):
    for summary_span in summary_spans:
        summary_tokens = summary_span.split()
        
        # For each token in the summary, find it in the review text
        for summary_token in summary_tokens:
            for i, review_token in enumerate(tokens):
                # Use fuzzy matching to handle minor variations
                if fuzz.ratio(summary_token.lower(), review_token.lower()) >= threshold:
                    labels[i] = 1
                    break  # Move to next summary token after first match

threshold = 80
parent_dir = pathlib.Path(__file__).resolve().parent

# load the csv file
annotated_data_path = parent_dir / "annotated_summary" / "summary_extraction_label_both.csv"
df_annotated = pd.read_csv(annotated_data_path)

ground_truths_1 = df_annotated["Ground Truth"].tolist()
ground_truths_2 = df_annotated["Ground Truth 2"].tolist()
review_texts = df_annotated["review_text"].tolist()

tie_breaker_results = []
jaccard_scores = []
annotator1_labels = []
annotator2_labels = []

# Initialize counters
true_positives = 0
false_positives = 0
false_negatives = 0

for gt_1, gt_2, review_text in tqdm(zip(ground_truths_1, ground_truths_2, review_texts), total=len(ground_truths_1), desc="Processing tie-breakers"):

    gt_1 = ast.literal_eval(gt_1)
    gt_2 = gt_2.replace("‘", "'").replace("’", "'")
    gt_2 = ast.literal_eval(gt_2)

    matches = []
    remaining = gt_2.copy()   # so we can remove items
    for a in gt_1:
        if not remaining:  # no more items to match
            break
        
        match, score, idx = process.extractOne(a, remaining, scorer=fuzz.partial_token_set_ratio)
        if score >= threshold:
            matches.append((a, match, score))
            remaining.pop(idx)  # remove matched item

    num_matches = len(matches)
    jaccard = num_matches / (len(gt_1) + len(gt_2) - num_matches)
    jaccard_scores.append(jaccard)

    if jaccard == 1.0:
        tie_breaker_results.append("agreed")
    else:
        tie_breaker_results.append("disagreed")
        
        
    true_positives += len(matches)
    false_positives += len(gt_1) - len(matches)
    false_negatives += len(gt_2) - len(matches)
    only_in_2 = len(remaining)
    
    
    # fill annotator labels for kappa calculation
    tokens = review_text.split()
    labels_1 = [0] * len(tokens)
    labels_2 = [0] * len(tokens)

    mark_extracted_tokens(labels_1, gt_1, tokens)
    mark_extracted_tokens(labels_2, gt_2, tokens)

    annotator1_labels.extend(labels_1)
    annotator2_labels.extend(labels_2)


df_annotated["tie_breaker_result"] = tie_breaker_results
output_path = parent_dir / "annotated_summary" / "summary_extraction_labels_with_tie_breaker.csv"
df_annotated.to_csv(output_path, index=False)

average_jaccard = sum(jaccard_scores) / len(jaccard_scores)
print(f"Average Jaccard similarity across all pairs: {average_jaccard:.4f}")

kappa = cohen_kappa_score(annotator1_labels, annotator2_labels)
print(f"Cohen's kappa: {kappa:.4f}")


# Calculate metrics
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
print(f"Overall Precision: {precision:.4f}")
print(f"Overall Recall: {recall:.4f}")
print(f"Overall F1 Score: {f1:.4f}")