import pytesseract
from PIL import Image
import subprocess
import easyocr
import pyocr
import pyocr.builders
import numpy as np
import spacy
import re
import cv2
import io
import os
import nltk
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from nltk.corpus import words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mutual_info_score
import language_tool_python
import argparse
import torch
from pathlib import Path
from pylanguagetool import api
from skimage.filters import threshold_local
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from eval import composite_score

parser = argparse.ArgumentParser()

parser.add_argument('--img_path', type=str, default=None)
parser.add_argument('--lang', type=str, default=None)
args = parser.parse_args()

def getDetectedText(img_path, lang='ru'):
	tools = pyocr.get_available_tools()
	if len(tools) == 0:
		raise RuntimeError("No OCR tool found")

	tool = tools[0]

	if lang == 'en':
		ocr_lang = 'eng'
	elif lang == 'ru':
		ocr_lang = 'rus'
	else:
		raise ValueError(f"Unsupported language: {lang}")

	img = Image.open(img_path)

	text = tool.image_to_string(
		img,
		lang=ocr_lang,
		builder=pyocr.builders.TextBuilder()
	)

	return text


def rotateAndSaveImage(img_path, angle):
	rotate_command = f"convert {img_path} -rotate {angle} -quality 100 {img_path}"
	subprocess.run(rotate_command, shell=True, check=True)
	identify_command = f"identify -format '%w %h' {img_path}"
	result = subprocess.run(identify_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	if result.stderr:
		print("Error in getting image dimensions:", result.stderr.decode())
		return None, None
	dimensions = result.stdout.decode().strip().split()
	width, height = map(int, dimensions)
	return width, height

def searchIterator(img_path, debug=True):
	results = {}
	rotated = -1
	for i in range(4):
		if i == 0:
			w, h = rotateAndSaveImage(img_path, 0)
		else:
			w, h = rotateAndSaveImage(img_path, 90)
		if h > w and rotated == -1:
			rotated = i
		for lang in ['ru', 'en']:
			text = getDetectedText(img_path, lang)
			special_symbols_removed, characters_removed, filtered_text = filterText(text, lang)
			composite_score_value, word_count, perplexity, errors, invalid_ratio = composite_score(filtered_text, lang)
			results[(lang, i)] = [composite_score_value, filtered_text, special_symbols_removed, characters_removed, word_count, errors, perplexity, invalid_ratio]
	rotateAndSaveImage(img_path, 90 + 90 * rotated)

	maximum_score = -float('inf')
	maximum_key = None
	i = 1
	for key, value in results.items():
		if i <= 4:
			imgName = os.path.basename(img_path)
			createJsonFile(results, key, imgName, rotated, i)
		i += 1
		score = value[0]
		if score > maximum_score:
			maximum_score = score
			maximum_key = key
	if debug:
		imgName = os.path.basename(img_path)
		createJsonFile(results, key, imgName, rotated, i)
		debugInfo(maximum_key, results)
	return maximum_key, results

def filterText(text, lang):
	if lang == 'en':
		filtered_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
	elif lang == 'ru':
		filtered_text = re.sub(r'[^а-яА-ЯёЁ0-9\s]', '', text)


	special_symbols_removed = len(re.findall(r'[^\w\s]', text))
	characters_removed = len(text) - len(filtered_text)
	return special_symbols_removed, characters_removed, filtered_text

def spellCheck(text):
	composite = composite_score(text)  # Используем composite_score
	return composite

def debugInfo(maximum_key, results):
	print("--------------------------")
	for key, contents in results.items():
		lang, orientation = key
		orientation = str(orientation)
		lang_dict = {'ru': 'RUSSIAN', 'en': 'ENGLISH', 'both': 'BOTH'}
		orientation_dict = {'0': 'ORIGINAL', '1': 'ROTATED90', '2': 'ROTATED180', '3': 'ROTATED270'}
		print(f"{orientation_dict[orientation]} image tested for {lang_dict[lang]}:")
		print(f"Composite Score: {contents[0]}")
		print("Number of SPECIAL SYMBOLS removed:", contents[2])
		print("Number of CHARACTERS removed:", contents[3])
		print("--------------------------")

def createJsonFile(results, maximum_key, image_name, was_rotated, inc):
	data_to_save = {
		"maximum_key": maximum_key,
		"was_rotated": was_rotated,
		"composite_score": results[maximum_key][0],
		"special_symbols_removed": results[maximum_key][2],
		"characters_removed": results[maximum_key][3],
		"detected_text": results[maximum_key][1],
		"word_count": results[maximum_key][4],
		"errors": results[maximum_key][5],
		"perplexity": results[maximum_key][6],
		"invalid_ratio": results[maximum_key][7]
	}

	base_name = os.path.basename(image_name)
	name_without_extension, _ = os.path.splitext(base_name)
	name_without_extension += str(inc)
	json_filename = f"{name_without_extension}.json"
	with open(json_filename, "w", encoding='utf-8') as json_file:
		json.dump(data_to_save, json_file, ensure_ascii=False, indent=4)

	print(f"Data saved to {json_filename}")

def parseJsonFile(file_name):
	try:
		with open(file_name, 'r') as json_file:
			data = json.load(json_file)
			return data
	except FileNotFoundError:
		print(f"File {file_name} not found.")
		return None
	except json.JSONDecodeError:
		print(f"Error decoding JSON from file {file_name}.")
		return None

def main():
	searchIterator(args.img_path)

if __name__ == "__main__":
	main()

