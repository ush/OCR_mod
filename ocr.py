import pytesseract
from PIL import Image
import subprocess
import easyocr
import pyocr
import pyocr.builders
import numpy as np
import nltk
import csv
import re
import cv2
import io
import os
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
from eval import grammatical_errors_score, word_count_score, language_model_score, composite_score, invalid_word_ratio
from grayscale import complex_preprocess_image, no_preprocess_image, apply_grayscale_script, get_new_path, get_file_name

parser = argparse.ArgumentParser(description='testing OCR and pytesseract')
parser.add_argument('--img_path', type=str, required=True, help='Path to the image file')
parser.add_argument('--prep', type=str, default='Y', help='Mode for non-preprocessing')
args = parser.parse_args()

english_stopwords = stopwords.words('english')
russian_stopwords = stopwords.words('russian')
	
# OCR Function
def read_text_from_image(image_path, prep, ocr_tool='pytesseract', lang='ru'):
	text = ""
	desk_img = ""
    
	if lang == 'en':
		ocr_lang = 'eng'
	elif lang == 'ru':
		ocr_lang = 'rus'

	out_path = get_new_path(image_path, "./Cleaned/")
	
	if prep == 'Y':	
		desk_img = complex_preprocess_image(image_path, out_path)
	else:
		desk_img = no_preprocess_image(image_path)
	
	if ocr_tool == 'pytesseract':
		img = Image.open(desk_img)
		text = pytesseract.image_to_string(img, lang=ocr_lang)

	elif ocr_tool == 'easyocr':
		reader = easyocr.Reader([lang])
		img = Image.open(desk_img)
		if img.mode == 'RGBA':
			img = img.convert('RGB')
		img_byte_arr = io.BytesIO()
		img.save(img_byte_arr, format='jpeg')
		img = img_byte_arr.getvalue()
		result = reader.readtext(img)
		text = "\n".join([detection[1] for detection in result])

	elif ocr_tool == 'pyocr':
		tools = pyocr.get_available_tools()
		if len(tools) == 0:
			raise RuntimeError("No OCR tool found")
		tool = tools[0]
		img = Image.open(desk_img)
		text = tool.image_to_string(img, lang=ocr_lang, builder=pyocr.builders.TextBuilder())

	else:
		raise ValueError(f"Unsupported OCR tool: {ocr_tool}")

	return text


def preprocess_text(text, lang = 'ru'):
    text = text.lower()
    
    # Убираем всё, кроме букв русского и английского языков и пробелов
    if lang == 'ru':
        text = re.sub(r'[^А-Яа-я\s]', '', text)
    elif lang == 'en':
        text = re.sub(r'[^A-Za-z\s]', '', text)

    # Заменяем переносы строк на пробелы
    text = text.replace("\n", " ") 

    # Убираем знаки препинания
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()

    # Токенизация и фильтрация стоп-слов в зависимости от языка
    if lang == 'ru':
        tokens = [word for word in word_tokenize(text, language='russian') if word not in russian_stopwords]
    elif lang == 'en':
        tokens = [word for word in word_tokenize(text, language='english') if word not in english_stopwords]
    
    return ' '.join(tokens)
	
# Function to print results neatly
def print_results(results):
	print("Comparison Results:")
	for metric, value in results.items():
		if isinstance(value, list):
			print(f"{metric}:")
			for i, v in enumerate(value):
				print(f"  Text {i+1}: {v}")
		else:
			print(f"{metric}: {value}")

def run_ocr_and_save_results(image_path, prep, ocr_tools, csv_filenames):
    languages = ['en', 'ru']
    results = {tool: {} for tool in ocr_tools}

    for tool in ocr_tools:
        best_score = None
        best_text = None
        best_lang = None
        best_details = None
        
        for lang in languages:
            text = preprocess_text(read_text_from_image(image_path, prep, tool, lang), lang)
            
            score, word_count, perplexity, errors, invalid_ratio = composite_score(text, lang)
            
            if best_score is None or score > best_score:
                best_score = score
                best_text = text
                best_lang = lang
                best_details = {
                    'word_count': word_count,
                    'perplexity': perplexity,
                    'errors': errors,
                    'invalid_ratio': invalid_ratio
                }
        
        results[tool] = {
            'best_lang': best_lang,
            'best_score': best_score,
            'best_text': best_text,
            **best_details
        }

    # Записываем результаты в CSV
    for tool, csv_filename in zip(ocr_tools, csv_filenames):
        with open(csv_filename, mode='a', encoding='utf-8') as csv_file:
            fieldnames = ['best_lang', 'best_score', 'word_count', 'perplexity', 'errors', 'invalid_ratio']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            writer.writerow({
                'best_lang': results[tool]['best_lang'],
                'best_score': results[tool]['best_score'],
                'word_count': results[tool]['word_count'],
                'perplexity': results[tool]['perplexity'],
                'errors': results[tool]['errors'],
                'invalid_ratio': results[tool]['invalid_ratio']
            })
def main():
	# Main comparison code
	image_path = args.img_path
	prep = args.prep
	image_name = get_file_name(image_path)

	# Считываем текст с помощью разных OCR инструментов
	ocr_tools = ['pytesseract', 'easyocr', 'pyocr']
	csv_filenames = ['pytesseract.csv', 'easyocr.csv', 'pyocr.csv']

	run_ocr_and_save_results(image_path, prep, ocr_tools, csv_filenames)
	return 0

if __name__ == "__main__":
    main()
