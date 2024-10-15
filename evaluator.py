import re
import os
import csv
import sys
import pyocr
import argparse
import numpy as np
import pyocr.builders
from PIL import Image
from os import listdir
from os.path import isfile, join
from eval import composite_score

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default=None)
parser.add_argument('--csv_path', type=str, default=None)
args = parser.parse_args()

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
    
def calculateMetricSaveToCSV(filepath, csv_path, lang, i):
    text = ""
    with open(filepath, 'r') as f:
        text = f.read()
    composite_score_value, word_count, perplexity, errors, invalid_ratio = composite_score(text, lang)
    with open(csv_path, mode='a+', encoding='utf-8') as csv_file:
        fieldnames = ['image_num', 'composite_score', 'word_count', 'perplexity', 'errors', 'invalid_ratio']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
        writer.writerow({
            'image_num': i,
            'composite_score': composite_score_value,
            'word_count': word_count,
            'perplexity': perplexity,
            'errors': errors,
            'invalid_ratio': invalid_ratio
        })

def progon(path, csv_path):
    allfiles = []
    for f in listdir(path):
        if f.endswith(".txt"):
            allfiles.append(join(path, f))
    
    allfiles = sorted_alphanumeric(allfiles)
    print(allfiles)
    
    i = 1
    for filepath in allfiles:
        if i <= 50:
            calculateMetricSaveToCSV(filepath, csv_path, "en", i)
        else:
            calculateMetricSaveToCSV(filepath, csv_path, "ru", i)
        i += 1
    return 0

def main():
    progon(args.path, args.csv_path)
    print("Texts generated!")

if __name__ == "__main__":
	main()
