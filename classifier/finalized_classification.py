
import sys
import pathlib
import pandas as pd
import json
from tqdm import tqdm
import re
import os

def natural_key(s: str):
    """
    Natural sort key: 'chunk2' < 'chunk10'
    Splits into text + integer parts.
    """
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


parent_dir = pathlib.Path(__file__).resolve().parent

# load the csv file
#annotated_data_path = parent_dir / "annotated_data_privacy" / "final_classified_with_appname_permissions.csv"
annotated_data_path = "/u/siddique-d1/Moghis/mobilerec/mobilerec_after_2021_03_18.csv"
root_dir = "/u/siddique-d1/Moghis/mobilerec"
ordered_folders = ["classifier_results_0_to_100k", "classifier_results_100k_to_200k", 
                   "classifier_results_200k_300k", "classifier_results_300k_to_400k",
                   "classifier_results_400k_500k", "classifier_results_500k_to_900k",
                   "classifier_results_900k_1.3m", "classifier_results_1.3m_to_1.7m",
                   "classifier_results_1.7m_to_2.1m", "classifier_results_2.1m_to_2.5m",
                   "classifier_results_2.5m_to_2.9m", "classifier_results_2.9m_to_3.3m",
                   "classifier_results_3.3m_to_3.7m", "classifier_results_3.7m_to_4.1m", 
                   "classifier_results_4.1m_to_end"]
df_annotated = pd.read_csv(annotated_data_path)
#df_annotated["date"] = pd.to_datetime(df_annotated["date"])
#df_annotated = df_annotated.sort_values(by="date", ascending=False)
#split_date = pd.Timestamp("2022-01-01")
#df_annotated = df_annotated[df_annotated["date"] >= split_date].reset_index(drop=True)


all_dfs = []

for folder in tqdm(ordered_folders, desc="Processing folders"):
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # list CSVs in this folder
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]

    # sort based on file names (natural order)
    csv_files_sorted = sorted(csv_files, key=natural_key)

    # load in that sorted order
    for fname in csv_files_sorted:
        fpath = os.path.join(folder_path, fname)
        df = pd.read_csv(fpath)
        # optional: keep provenance
        df["__folder"] = folder
        df["__file"] = fname
        all_dfs.append(df)

finalized_df = pd.concat(all_dfs, ignore_index=True)

# unprocessed data
df_annotated_unprocessed = df_annotated.iloc[finalized_df.shape[0]:].copy()
# save unprocessed data
unprocessed_save_path = f"{root_dir}/all_unprocessed.csv"
df_annotated_unprocessed.to_csv(unprocessed_save_path, index=False)

df_annotated = df_annotated.iloc[:finalized_df.shape[0]].copy()
df_annotated["label"] = finalized_df["label"]
df_annotated = df_annotated[df_annotated["label"] == "yes"]
save_path = f"{root_dir}/all_final_classified.csv"
df_annotated.to_csv(save_path, index=False)