
import sys
import pathlib
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
import ast
from rapidfuzz import process, fuzz
from collections import Counter
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from promptLLM import open_route_ai_models

model_id = "meta-llama_llama-3.3-70b-instruct"
provider =  "openai"
threshold = 80

parent_dir = pathlib.Path(__file__).resolve().parent

# load the csv file
data_path = parent_dir.parent / "classifier" / "annotated_data_privacy" / "final_classified_with_appname_permissions.csv"
df_data = pd.read_csv(data_path)

# keep only the positive classifications
df_data = df_data[df_data["classification"] == 1].reset_index(drop=True)

# read the txt file
path = parent_dir / "label_generator_prompt.txt"
with open(path, 'r') as file:
    prompt_txt = file.read()
    
annotated_data_path = parent_dir / "annotated_summary" / "summary_extraction_labels_finalized.csv"
if annotated_data_path.exists():
    df_annotated = pd.read_csv(annotated_data_path)
else:
    df_annotated = None

results = []
# loop through the csv file and replace the [$NAME$] and [$REVIEW$] in the prompt
for index, row in tqdm(df_data.iterrows(), total=df_data.shape[0], desc="Processing rows"):
    prompt = prompt_txt.replace("[$NAME$]", row["app_name"])
    prompt = prompt.replace("[$REVIEW$]", row["review"])

    output = open_route_ai_models(
        prompt,
        model_id,
        json_output=True,
        provider=provider
    )
    
    try:
        
        json_part = output[output.find("{"): output.rfind("}") + 1].strip().lower()
        
        summaries = json.loads(json_part)["privacy/security summaries"]

        results.append({"app_name": row["app_name"], "review_text": row["review"], "summaries": summaries})

    except Exception as e:
        print(f"Error parsing JSON for row {index}: {e}\nResponse was: {output}. Skipping.")
        results.append({"review_text": row["review"], "summaries": []})
        continue

df_results = pd.DataFrame(results)

# create output directory if it doesn't exist
output_dir = parent_dir / "output" 
output_dir.mkdir(parents=True, exist_ok=True)

df_results.to_csv(output_dir / f"{model_id.replace('/', '_')}.csv", index=False)

def token_f1(pred_spans, gt_spans):
    pred_tokens = Counter(" ".join(pred_spans).lower().split())
    gt_tokens = Counter(" ".join(gt_spans).lower().split())
    common = sum((pred_tokens & gt_tokens).values())
    if common == 0:
        return 0.0, 0.0, 0.0
    p = common / sum(pred_tokens.values())
    r = common / sum(gt_tokens.values())
    return p, r, 2 * p * r / (p + r)


if df_annotated is not None:
    jaccard_scores = []
    Precision = []
    Recall = []
    F1 = []
    Dice_Score = []
    Token_Precision = []
    Token_Recall = []
    Token_F1 = []

    # human annotated summaries
    final_ground_truth = df_annotated["final_ground_truth"].tolist()
    predicted_summaries = df_results["summaries"].tolist()
    
    for gt, pred in zip(final_ground_truth, predicted_summaries):
        
        gt = ast.literal_eval(gt)
        if isinstance(pred, str):
            pred = ast.literal_eval(pred)
        
        matches = []
        remaining = gt.copy()   # so we can remove items
        for a in pred:
            if not remaining:  # no more items to match
                break
            
            match, score, idx = process.extractOne(a, remaining, scorer=fuzz.partial_token_set_ratio)
            if score >= threshold:
                matches.append((a, match, score))
                remaining.pop(idx)  # remove matched item

        num_matches = len(matches)
        jaccard = num_matches / (len(gt) + len(pred) - num_matches)
        jaccard_scores.append(jaccard)
        
        Precision.append(num_matches / len(pred) if len(pred) > 0 else 0)
        Recall.append(num_matches / len(gt) if len(gt) > 0 else 0)
        
        if Precision[-1] + Recall[-1] > 0:
            F1.append(2 * Precision[-1] * Recall[-1] / (Precision[-1] + Recall[-1]))
            Dice_Score.append(2 * num_matches / (len(gt) + len(pred)))
        else:
            F1.append(0)
            Dice_Score.append(0)

        tp, tr, tf = token_f1(pred, gt)
        Token_Precision.append(tp)
        Token_Recall.append(tr)
        Token_F1.append(tf)


average_jaccard = sum(jaccard_scores) / len(jaccard_scores)
print(f"Average Jaccard similarity across all pairs: {average_jaccard:.4f}")
print(f"Average Precision: {sum(Precision) / len(Precision):.4f}")
print(f"Average Recall: {sum(Recall) / len(Recall):.4f}")
print(f"Average F1 Score: {sum(F1) / len(F1):.4f}")
print(f"Average Dice Score: {sum(Dice_Score) / len(Dice_Score):.4f}")
print(f"\n--- Token-level F1 (SQuAD-style) ---")
print(f"Token Precision: {sum(Token_Precision) / len(Token_Precision):.4f}")
print(f"Token Recall:    {sum(Token_Recall) / len(Token_Recall):.4f}")
print(f"Token F1:        {sum(Token_F1) / len(Token_F1):.4f}")