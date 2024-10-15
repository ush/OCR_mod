import re
import os
import csv
import sys
import pyocr
import argparse
import numpy as np
import pandas as pd
import pyocr.builders
from PIL import Image
from os import listdir
from os.path import isfile, join
from eval import composite_score_tech

parser = argparse.ArgumentParser()

parser.add_argument('--orig_path', type=str, default=None)
parser.add_argument('--taint_path', type=str, default=None)
parser.add_argument('--true_path', type=str, default=None)
args = parser.parse_args()

def progon(orig_path, taint_path, true_path):
    df_orig = pd.read_csv(orig_path, sep=',', names = ['img_num', 'composite', 'word_count', 'perplexity', 'errors', 'invalid_ratio'])
    df_taint = pd.read_csv(taint_path, sep=',', names = ['img_num', 'composite', 'word_count', 'perplexity', 'errors', 'invalid_ratio'])
    df_true = pd.read_csv(true_path, sep=',', names = ['img_num', 'composite', 'word_count', 'perplexity', 'errors', 'invalid_ratio'])
    
    df_orig.reset_index(drop=True, inplace=True)
    df_taint.reset_index(drop=True, inplace=True)
    df_true.reset_index(drop=True, inplace=True)
    
    df_comp = pd.concat([df_orig['composite'], df_taint['composite'], df_true['composite']], axis=1)
    df_comp.columns = ['orig', 'taint', 'true']
    
    df_comp['comp_orig_taint'] = np.where(df_comp['orig'] > df_comp['taint'], 1, 0)
    df_comp['comp_true_orig'] = np.where(df_comp['true'] > df_comp['orig'], 1, 0)
    df_comp['comp_true_taint'] = np.where(df_comp['true'] > df_comp['taint'], 1, 0)
    
    print("Percentage of cases when original is better than tainted: ", np.sum(df_comp['comp_orig_taint']), "%")
    print("Percentage of cases when true is better than original: ", np.sum(df_comp['comp_true_orig']), "%")
    print("Percentage of cases when true is better than tainted: ", np.sum(df_comp['comp_true_taint']), "%")
    

def main():
    progon(args.orig_path, args.taint_path, args.true_path)

if __name__ == "__main__":
	main()
