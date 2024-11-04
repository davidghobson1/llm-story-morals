import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer, util
import evaluate
import torch

import scipy
from scipy.stats import mannwhitneyu

import os
import argparse
from typing import Callable

rouge = evaluate.load('rouge')
bertscore = evaluate.load('bertscore')

# Compute rouge-L score given the predictions and references
def rouge_L(preds: pd.Series, refs: pd.Series) -> list[float]:
    return rouge.compute(predictions=preds, references=refs, use_aggregator=False)['rougeL']

# Compute BERTScore (F1) given the predictions and references
def bertscore_f1(preds: pd.Series, refs: pd.Series) -> list[float]:
    # this model has the best performance on Huggingface's BERTScore leaderboard, in terms of correlation with humans
    # it is a bit slow though
    return bertscore.compute(predictions=preds, references=refs, model_type="microsoft/deberta-xlarge-mnli", lang="en")['f1']

# Compute the pairwise cosine similarity between sentence transformer embeddings given the predictions, references, and the model name
def sbert_cos_sim(preds: pd.Series, refs: pd.Series, model_name: str = "nli-mpnet-base-v2") -> list[float]:
    model = SentenceTransformer(model_name)
    v1 = model.encode(preds, convert_to_tensor=True)
    v2 = model.encode(refs, convert_to_tensor=True)
    return util.pairwise_cos_sim(v1, v2)

def compute_intra_scores(df: pd.DataFrame, score_func: Callable[pd.Series, pd.Series], **kwargs) -> torch.Tensor:
    cols = df.columns
    num_combos = scipy.special.comb(len(cols), 2).astype(int)
    scores = torch.zeros(df.shape[0], num_combos)
    
    k = 0
    for i in range(len(cols)-1):
        refs = df[cols[i]].tolist()
        for j in range(i+1, len(cols)):
            preds = df[cols[j]].tolist()
            scores[:, k] = torch.Tensor(score_func(preds, refs, **kwargs))
            k += 1
    return scores.flatten()

def compute_inter_scores(df1: pd.DataFrame, df2: pd.DataFrame, score_func: Callable[pd.Series, pd.Series], **kwargs) -> torch.Tensor:
    if df1.shape[0] != df2.shape[0]:
        print("Error: size mismatch")
        return
    cols1, cols2 = df1.columns, df2.columns
    num_combos = len(cols1)*len(cols2)
    scores = torch.zeros(df1.shape[0], num_combos)

    k = 0
    for i in range(len(cols1)):
        # print(i)
        refs = df1[cols1[i]].tolist()
        for j in range(len(cols2)):
            # print("\t", j)
            preds = df2[cols2[j]].tolist()
            scores[:, k] = torch.Tensor(score_func(preds, refs, **kwargs))
            k += 1
    return scores.flatten()

# df_subset1 is all the human morals, df_subset2 is all the GPT morals
# Compute the intra and inter-dataframe scores between the two dataframes
# score_func is a dictionary of the scoring functions to be applied (keys are the human-readable names for the functions)
def get_distributions(df_subset1: pd.DataFrame, df_subset2: pd.DataFrame, score_funcs: dict[str:dict]) -> dict[str:dict[str:torch.Tensor]]:
    data = {'1-1': dict(), '2-2': dict(), '1-2': dict()}
    for score_name, score_func in score_funcs.items():
        print(f"\t{score_name}")

        # Compute the intra-dataframe scores
        if score_name == 'BERTScore':
            print("\t\tComputing intra-dataframe similarities... (this may take several minutes; BERTScore might be quite slow)")
        else:
            print("\t\tComputing intra-dataframe similarities...")
        data['1-1'][score_name] = compute_intra_scores(df_subset1, score_func['func'], **score_func['kwargs'])
        data['2-2'][score_name] = compute_intra_scores(df_subset2, score_func['func'], **score_func['kwargs'])
        
        # Compute the inter-dataframe scores
        if score_name == 'BERTScore':
            print("\t\tComputing inter-dataframe similarities... (this may take ~30 minutes; BERTScore might be quite slow)")
        else:
            print("\t\tComputing inter-dataframe similarities...")
        data['1-2'][score_name] = compute_inter_scores(df_subset1, df_subset2, score_func['func'], **score_func['kwargs'])
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated metric validation for comparing human and GPT moral responses')
    parser.add_argument('-cate', "--category_name", help="category name by which human and GPT responses will be compared. Valid values are 'moral', 'moral+', 'moral-', 'central_topic'. Default is 'moral'", default='moral')
    parser.add_argument('-output_filepath', "--output_filepath", help="filepath where the .csv output file will be saved. The output file will have the name {cate}_automated_validation_table.csv. Default is ../outputs", default='../outputs')

    args = parser.parse_args()
    cate = args.category_name
    output_filepath = args.output_filepath.replace("{cate}", cate)
    output_file = os.path.join(output_filepath, f"{cate}_automated_validation_table.csv")

    # Make the output directory if it doesn't exist
    if not os.path.exists(output_filepath):
        os.mkdir(output_filepath)

    results_dir = "/Users/david/Documents/Coding/GitHub/llm-story-morals/data/validation/moral_annotations"

    # Load the response data
    df_eng = pd.read_csv(os.path.join(results_dir, "human_responses_english.csv"), index_col='index').fillna("None")
    df_gpt_eng = pd.read_csv(os.path.join(results_dir, "gpt_responses_english.csv"), index_col='index').fillna("None")
    df_chin = pd.read_csv(os.path.join(results_dir, "human_responses_mandarin.csv"), index_col='index').fillna("None")
    df_gpt_chin = pd.read_csv(os.path.join(results_dir, "gpt_responses_mandarin.csv"), index_col='index').fillna("None")

    # Get the relevant columns for the given category
    human_eng_annotators, human_chin_annotators = ['AL', 'AS', 'AS2', 'AZ', 'EA', 'NW'], ['JP', 'VX', 'YN', 'JY']
    gpt_eng_annotators, gpt_chin_annotators = ['0', '1', '2', '3', '4', '5'], ['0', '1', '2', '3']
    
    human_eng_cols = [f"{ann}_{cate}" for ann in human_eng_annotators]
    gpt_eng_cols = [f"{ann}_{cate}" for ann in gpt_eng_annotators]
    human_chin_cols = [f"{ann}_{cate}" for ann in human_chin_annotators]
    gpt_chin_cols = [f"{ann}_{cate}" for ann in gpt_chin_annotators]

    print(f"Computing the automated metrics for: {cate}")

    # Name and specify the scoring functions (as well as their kwargs)
    score_funcs = {
        'Rouge-L': {'func': rouge_L, 'kwargs': {}},
        'BERTScore': {'func': bertscore_f1, 'kwargs': {}},
        'GloVe': {'func': sbert_cos_sim, 'kwargs': {'model_name': "average_word_embeddings_glove.6B.300d"}},
        'STSb-MPNet': {'func': sbert_cos_sim, 'kwargs': {'model_name': "stsb-mpnet-base-v2"}},
        'NLI-MPNet': {'func': sbert_cos_sim, 'kwargs': {'model_name': "nli-mpnet-base-v2"}}
    }

    # Compute the English and Mandarin distibutions
    print("English distribution calculation")
    eng_dists = get_distributions(df_eng[human_eng_cols], df_gpt_eng[gpt_eng_cols], score_funcs)
    print()
    print("Mandarin distribution calculation")
    chin_dists = get_distributions(df_chin[human_chin_cols], df_gpt_chin[gpt_chin_cols], score_funcs)

    # Combine distributions together
    print()
    print("Combining distributions")
    total_dists = {'1-1': dict(), '2-2': dict(), '1-2': dict()}
    for dist in total_dists:
        for score_name in score_funcs:
            total_dists[dist][score_name] = torch.concat((eng_dists[dist][score_name], chin_dists[dist][score_name]))

    # Rename the index
    subset_names = ['human', 'GPT']
    idx_map = {
        '1-1': f"{subset_names[0]}-{subset_names[0]}",
        '2-2': f"{subset_names[1]}-{subset_names[1]}",
        '1-2': f"{subset_names[0]}-{subset_names[1]}",
    }
    total_dists = {idx_map[idx]: total_dists[idx] for idx in ['1-1', '1-2', '2-2']}

    # Compute the medians
    medians = dict()
    for subset in total_dists:
        medians[subset] = dict()
        for metric in total_dists[subset]:
            medians[subset][metric] = total_dists[subset][metric].median().item()
    df_medians = pd.DataFrame(medians)*100

    # Compute the p-values between human-human and human-GPT distributions
    pvals = dict()
    for metric in total_dists['human-human']:
        pvals[metric] = mannwhitneyu(total_dists['human-human'][metric], total_dists['human-GPT'][metric]).pvalue
    df_medians['p-value'] = pvals

    # Save the file
    df_medians.round(2).to_csv(output_file)
    print(f"Results saved to {output_file}")