
import sys
import pathlib
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from promptLLM import open_route_ai_models


model_id = "meta-llama/llama-3.3-70b-instruct"
provider = "groq"
prompt_type = "few-shot-permissions-rules"


parent_dir = pathlib.Path(__file__).resolve().parent

# load the csv file
#annotated_data_path = parent_dir / "annotated_data_privacy" / "final_classified_with_appname_permissions.csv"
annotated_data_path = "/u/siddique-d1/Moghis/mobilerec/mobilerec_after_2021_03_18.csv"
df_annotated = pd.read_csv(annotated_data_path)
#df_annotated["date"] = pd.to_datetime(df_annotated["date"])
#df_annotated = df_annotated.sort_values(by="date", ascending=False)
#split_date = pd.Timestamp("2022-01-01")
#df_annotated = df_annotated[df_annotated["date"] >= split_date]
#df_annotated = df_annotated[4100000:] 

# read the txt file
naive_path = parent_dir / "llm_prompts" / f"{prompt_type}.txt"
with open(naive_path, 'r') as file:
    naive_prompt = file.read()
    
output_dir = pathlib.Path("/u/siddique-d1/Moghis/mobilerec/classifier_results_4.1m_to_end")
output_dir.mkdir(parents=True, exist_ok=True)

existing_chunks = sorted(output_dir.glob("classifer_results_chunk*.csv"))

processed_rows = 0
if existing_chunks:
    for chunk_path in existing_chunks:
        try:
            tmp_df = pd.read_csv(chunk_path)
            processed_rows += len(tmp_df)
        except Exception as e:
            print(f"Warning: could not read {chunk_path}: {e}")

    # Start new chunks after the last existing one
    chunk_id = len(existing_chunks)
else:
    chunk_id = 0
    
print(f"Found {len(existing_chunks)} existing chunk files.")
print(f"Already processed rows (according to saved chunks): {processed_rows}")

results = []
save_every = 10000

# loop through the csv file and replace the [$NAME$] and [$REVIEW$] in the prompt
for index, (_, row) in tqdm(enumerate(df_annotated.iterrows()), total=df_annotated.shape[0], desc="Processing rows"):
    
    if index < processed_rows:
        continue
    
    try:
        prompt = naive_prompt.replace("[$NAME$]", row["app_name"])
        prompt = prompt.replace("[$REVIEW$]", row["review"])
        prompt = prompt.replace("[$PERMISSIONS$]", row["permissions"].replace("This app has access to:\n", "").replace("\n", ", "))

        output = open_route_ai_models(
            prompt,
            model_id,
            json_output=True,
            provider=provider,
        )
        
        json_part = output[output.find("{"): output.rfind("}") + 1].strip().lower()
        
        label = json.loads(json_part)["privacy/security related"]
        summaries = json.loads(json_part)["privacy/security summaries"]
        
        results.append({
            "app_name": row["app_name"],
            "permissions": row["permissions"],
            "review_text": row["review"],
            "label": label,
            "summaries": summaries,
            # Optional: keep track of original position in df
            "row_idx": index,
        })
        
    except Exception as e:
        print(f"Error parsing JSON for row {index}: {e}\nResponse was: {output}. Skipping.")
        results.append({
            "app_name": row.get("app_name", ""),
            "permissions": row.get("permissions", ""),
            "review_text": row.get("review", ""),
            "label": "n/a",
            "summaries": [],
            "row_idx": index,
        })
        continue
    
    # -------------- SAVE EVERY save_every ROWS ---------------- #
    if len(results) >= save_every:
        df_temp = pd.DataFrame(results)
        chunk_path = output_dir / f"classifer_results_chunk{chunk_id}.csv"
        df_temp.to_csv(chunk_path, index=False)
        print(f"Saved {len(results)} rows to {chunk_path}")
        
        results = []    # clear buffer
        chunk_id += 1
    # ------------------------------------------------------- #

df_results = pd.DataFrame(results)

# create output directory if it doesn't exist



# SAVE REMAINING
if len(results) > 0:
    df_temp = pd.DataFrame(results)
    chunk_path = output_dir / f"classifer_results_chunk{chunk_id}.csv"
    df_temp.to_csv(chunk_path, index=False)
    print(f"Saved final {len(results)} rows to {chunk_path}")