# -*- coding: utf-8 -*-
"""
@File        :   merge_results.py
@Author      :   Shi Chen
@Time        :   2024/10/11
@Description :   Merge inference results from Heron benchmark.
"""

import argparse
import glob
import os
import jsonlines
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, help="Directory to the results files.")
    parser.add_argument("--outputs-dir", type=str, help="Directory to the output files.")
    parser.add_argument("--categories", type=str, nargs='+', choices=["complex", "conv", "detail"],
                        help="Question categories to merge.")

    return parser.parse_args()

def run(args):
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    
    case_infos = {os.path.splitext(os.path.split(result_path)[1])[0]: result_path \
        for result_path in sorted(glob.glob(os.path.join(args.results_dir, "*.jsonl")))}

    for img_id in tqdm(range(1, 22)):
        img_name = '%03d.jpg' % img_id
        output_path = os.path.join(args.outputs_dir, '%03d.csv' % img_id)
        result_infos = {"question_id": [], 
                        "question": [], 
                        "category": [],} 
                        #"image_category": []}
        result_infos = {"question": []}
        b_first = True
        for case_name, result_path in case_infos.items():
            result_infos[case_name] = []
            with open(result_path) as f:
                for line in jsonlines.Reader(f):
                    if line["image"] == img_name and line["category"] in args.categories:
                        result_infos[case_name].append(line["answer"].strip())
                        if b_first:
                            #result_infos["question_id"].append(line["question_id"])
                            result_infos["question"].append(line["question"])
                            #result_infos["category"].append(line["category"])
                            #result_infos["image_category"].append(line["image_category"])
            b_first = False

        df = pd.DataFrame(result_infos)
        df.to_csv(output_path, index=False, sep=",")        

if __name__ == "__main__":
    args = parse_args()
    run(args)
