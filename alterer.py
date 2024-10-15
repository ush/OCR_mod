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

parser.add_argument('--csv_path', type=str, default=None)
args = parser.parse_args()

def progon(csv_path):
    df = pd.read_csv(csv_path, sep=',', names = ['img_num', 'composite', 'word_count', 'perplexity', 'errors', 'invalid_ratio'])
    
    composite = list(map(composite_score_tech, df['word_count'], df['perplexity'], df['errors'], df['invalid_ratio']))
    print(composite)
    print(len(composite))
    
    df['composite'] = composite
    df.to_csv(csv_path, index=False, header=False) 

def main():
    progon(args.csv_path)

if __name__ == "__main__":
	main()
