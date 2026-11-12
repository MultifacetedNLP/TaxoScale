
import sys
import pathlib
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from promptLLM import open_route_ai_models

model_ids = [("meta-llama/llama-3.3-70b-instruct", "groq"), ("meta-llama/llama-3.1-8b-instruct", "groq"),
            ("google/gemma-3-27b-it", "novita/bf16"), ("google/gemma-3-4b-it", "deepinfra/bf16"),
            ("google/gemma-3-12b-it", "crusoe/bf16"),
            ("mistralai/mixtral-8x7b-instruct", "deepinfra/fp8"), ("mistralai/mixtral-8x22b-instruct", "mistral"),
            ("mistralai/mistral-7b-instruct", "deepinfra/bf16"),
            ("deepseek/deepseek-r1-distill-llama-70b", "chutes/bf16"),
            ("deepseek/deepseek-r1-distill-qwen-32b", "deepinfra/fp8")] # we can test "cloudflare" later on

#model_ids = [("deepseek/deepseek-r1-distill-qwen-32b", "cloudflare")]

#model_ids = [("meta-llama/llama-3.3-70b-instruct", "groq")]


all_prompt_types = ["0-shot-naive", "0-shot-rules", "0-shot-permissions", "0-shot-permissions-rules", 
                    "few-shot", "few-shot-rules", "few-shot-permissions", "few-shot-permissions-rules"]

#all_prompt_types = ["few-shot-permissions-rules"]


parent_dir = pathlib.Path(__file__).resolve().parent

# load the csv file
annotated_data_path = parent_dir / "annotated_data_privacy" / "final_classified_with_appname_permissions.csv"
df_annotated = pd.read_csv(annotated_data_path)

for model_id, provider in model_ids:

    for prompt_type in all_prompt_types:

        # read the txt file
        naive_path = parent_dir / "llm_prompts" / f"{prompt_type}.txt"
        with open(naive_path, 'r') as file:
            naive_prompt = file.read()

        results = []
        # loop through the csv file and replace the [$NAME$] and [$REVIEW$] in the prompt
        for index, row in tqdm(df_annotated.iterrows(), total=df_annotated.shape[0], desc="Processing rows"):
            prompt = naive_prompt.replace("[$NAME$]", row["app_name"])
            prompt = prompt.replace("[$REVIEW$]", row["review"])
            prompt = prompt.replace("[$PERMISSIONS$]", row["permissions"].replace("This app has access to:\n", "").replace("\n", ", "))

            output = open_route_ai_models(
                prompt,
                model_id,
                json_output=True,
                provider=provider
            )
            
            try:
                
                json_part = output[output.find("{"): output.rfind("}") + 1].strip().lower()
                
                label = json.loads(json_part)["privacy/security related"]
                summaries = json.loads(json_part)["privacy/security summaries"]
                
                results.append({"app_name": row["app_name"], "review_text": row["review"], "label": label, "summaries": summaries, "classification": row["classification"]})
                
            except Exception as e:
                print(f"Error parsing JSON for row {index}: {e}\nResponse was: {output}. Skipping.")
                results.append({"review_text": row["review"], "label": "n/a", "summaries": [], "classification": row["classification"]})
                continue

        df_results = pd.DataFrame(results)

        # create output directory if it doesn't exist
        output_dir = parent_dir / "output" / f"{model_id.replace('/', '_')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        df_results.to_csv(output_dir / f"results_{prompt_type}.csv", index=False)

        predicted_labels = df_results["label"].tolist()
        # convert "yes" to 1 and "no or n/a" to 0
        predicted_labels = [1 if label == "yes" else 0 for label in predicted_labels]
        true_labels = df_annotated["classification"].tolist()

        # calculating F1 score
        print(f"Calculating scores for {prompt_type} and model {model_id}")

        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        print(f"F1 Score - weighted: {f1}")
        
        # sklearn precision
        precision = precision_score(true_labels, predicted_labels, pos_label=1)
        print(f"Precision Score: {precision}")

        # sklearn recall
        recall = recall_score(true_labels, predicted_labels, pos_label=1)
        print(f"Recall Score: {recall}")

        report = classification_report(true_labels, predicted_labels)
        print(f"Classification Report:\n{report}")

        # save the scores to a text file
        with open(output_dir / f"scores_{prompt_type}.txt", 'w') as file:
            file.write(f"F1 Score - weighted: {f1}\n")
            file.write(f"Precision Score: {precision}\n")
            file.write(f"Recall Score: {recall}\n")
            file.write(f"Classification Report:\n{report}\n")