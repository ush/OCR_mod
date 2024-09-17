import pytesseract
from PIL import Image
import subprocess
import easyocr
import pyocr
import pyocr.builders
import numpy as np
import nltk
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
from eval import grammatical_errors_score, word_count_score, language_model_score, composite_score

parser = argparse.ArgumentParser(description='testing OCR and pytesseract')
parser.add_argument('--img_path', type=str, required=True, help='Path to the image file')
parser.add_argument('--lang', type=str, default='ru', help='Language for OCR (default: ru)')
parser.add_argument('--prep', type=str, default='Y', help='Mode for non-preprocessing')
args = parser.parse_args()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('udhr')
russian_stopwords = set(stopwords.words('russian'))

def get_file_name(file_path):
	file_name = Path(file_path).name
	return file_name

def get_new_path(file_path, dir_path):
	file_name = get_file_name(file_path)
	os.makedirs(dir_path, exist_ok = True)
	new_image_path = dir_path + file_name
	return new_image_path

def apply_grayscale_script(image_path):
	new_image_path = get_new_path(image_path, "./Cleaned/")
	subprocess.run(["bash", "textcleaner", "-g", "-e", "normalize", "-f", "25", "-o", "20", "-s", "1", image_path, new_image_path])
	return new_image_path


def complex_preprocess_image(image_path, scale_factor=2):
	# Read the image in color
	image = cv2.imread(image_path)

	# Convert to grayscale for some operations but keep the original
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Апскейлинг изображения
	upscaled_image = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    	
    	# Enhance contrast
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	enhanced_image = clahe.apply(upscaled_image)
	
	new_image_path = get_new_path(image_path, "./Enhanced/")
	cv2.imwrite(new_image_path, enhanced_image)
	
	final_image_path = apply_grayscale_script(new_image_path)
	
	final_image = cv2.imread(final_image_path)

	return final_image
	
def no_preprocess_image(image_path):
	image = cv2.imread(image_path)
	return image
	
def simple_preprocess_image(image_path, name):
	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	image = cv2.equalizeHist(image)
	image = cv2.medianBlur(image, 3)
	_, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	cv2.imwrite("./" + name + ".jpg", image)
	return image

# OCR Function
def read_text_from_image(image_path, prep, ocr_tool='pytesseract', lang='rus'):
	text = ""
	desk_img = ""

	if prep == 'Y':	
		desk_img = complex_preprocess_image(image_path)
	else:
		desk_img = no_preprocess_image(image_path)
	
	if ocr_tool == 'pytesseract':
		img = Image.fromarray(desk_img)
		text = pytesseract.image_to_string(img, lang=lang)

	elif ocr_tool == 'easyocr':
		reader = easyocr.Reader([lang])
		img = Image.fromarray(desk_img)
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
		img = Image.fromarray(desk_img)
		text = tool.image_to_string(img, lang=lang, builder=pyocr.builders.TextBuilder())

	else:
		raise ValueError(f"Unsupported OCR tool: {ocr_tool}")

	return text

# Load Russian stopwords
russian_stopwords = set(stopwords.words('russian'))

# Load Russian lexicon from file, assuming it might be encoded in ISO-8859-1 or CP1251
def load_russian_lexicon(file_path):
	try:
		with open(file_path, 'r', encoding='utf-8') as file:
			return set(word.strip().lower() for word in file)
	except UnicodeDecodeError:
		with open(file_path, 'r', encoding='cp1251') as file:
			return set(word.strip().lower() for word in file)

russian_lexicon = load_russian_lexicon('russian.txt')

# Function to preprocess text for Russian
def preprocess_text(text):
	text = text.lower()
	text = re.sub(r'[^А-Яа-яA-Za-z\s]', '', text)
	text = text.replace("\n", " ") 
	text = text.translate(str.maketrans('', '', string.punctuation))
	text = text.strip()
	tokens = [word for word in word_tokenize(text, language='russian') if word not in russian_stopwords]
	return ' '.join(tokens)

# Comprehensive function for comparison
def compare_texts(text1, text2, text3):
	texts = [preprocess_text(text1), preprocess_text(text2), preprocess_text(text3)]
	results = {}
	
	# Grammatical errors score
	results['Grammatical Errors'] = [grammatical_errors_score(text) for text in texts]
	
	results['Language Model Perplexity'] = [language_model_score(text) for text in texts]
	
	results['Word Count'] = [word_count_score(text) for text in texts]
	
	results['Composite Score'] = [composite_score(text)[0] for text in texts]
	
	return results
	
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

def main():
	# Main comparison code
	image_path = args.img_path
	lang = args.lang
	prep = args.prep
	image_name = get_file_name(image_path)

	# Считываем текст с помощью разных OCR инструментов
	ocr_tools = ['pytesseract', 'easyocr', 'pyocr']
	texts = {}
	for tool in ocr_tools:
		language = lang
		match tool:
			case 'pytesseract':
				language = 'rus'
			case 'easyocr':
				language = 'ru'
			case 'pyocr':
				language = 'rus'
		texts.update({tool: preprocess_text(read_text_from_image(image_path, prep, tool, language))})

	# Сравниваем результаты по метрике
	results = compare_texts(texts['pytesseract'], texts['easyocr'], texts['pyocr'])
	composite = results['Composite Score'][0]
	word = results['Word Count']
	perplexity = results['Language Model Perplexity']
	errors = results['Grammatical Errors']

	if prep == 'Y':
		with open('pytesseract.csv', 'a') as f:
			f.write(str(composite[0]) + "," + str(word[0]) + "," + str(perplexity[0]) + "," + str(errors[0]) + "\n")
			#f.write(image_name + "\n")
			#f.write(texts['pytesseract'] + "\n")
	
		with open('easyocr.csv', 'a') as f:
			f.write(str(composite[1]) + "," + str(word[1]) + "," + str(perplexity[1]) + "," + str(errors[1]) + "\n")
			#f.write(image_name + "\n")
			#f.write(texts['easyocr'] + "\n")
	
		with open('pyocr.csv', 'a') as f:
	    		f.write(str(composite[2]) + "," + str(word[2]) + "," + str(perplexity[2]) + "," + str(errors[2]) + "\n")
	    		#f.write(image_name + "\n")
	    		#f.write(texts['pyocr'] + "\n")
	else:
		with open('raw_pytesseract.csv', 'a') as f:
			f.write(str(composite[0]) + "," + str(word[0]) + "," + str(perplexity[0]) + "," + str(errors[0]) + "\n")
			#f.write(image_name + "\n")
			#f.write(texts['pytesseract'] + "\n")
	
		with open('raw_easyocr.csv', 'a') as f:
			f.write(str(composite[1]) + "," + str(word[1]) + "," + str(perplexity[1]) + "," + str(errors[1]) + "\n")
			#f.write(image_name + "\n")
			#f.write(texts['easyocr'] + "\n")
	
		with open('raw_pyocr.csv', 'a') as f:
	    		f.write(str(composite[2]) + "," + str(word[2]) + "," + str(perplexity[2]) + "," + str(errors[2]) + "\n")
		

if __name__ == "__main__":
    main()
