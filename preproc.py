from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
import Levenshtein
import subprocess
import tempfile
import argparse
import spacy
import nltk
import json
import cv2
import re
import os
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nlp_en = spacy.load("en_core_web_sm")
nlp_ru = spacy.load("ru_core_news_sm")
spell_en = SpellChecker(language='en')
spell_ru = SpellChecker(language='ru')

parser = argparse.ArgumentParser()

parser.add_argument('-imgPath', type=str, default=None)
parser.add_argument('-lang', type=str, default=None)
args = parser.parse_args()

def getDetectedText(imgPath, lang = 'both'):
	match lang:
		case "en":
			lang = "eng"
		case "ru":
			lang = "rus"
		case "both":
			lang = "eng+rus"
	cmd = ['tesseract', imgPath, '-', '-l', lang]
	with tempfile.TemporaryFile() as tempf:
		proc = subprocess.Popen(cmd, stdout=tempf)
		proc.wait()
		tempf.seek(0)
		s = tempf.read().decode("utf-8")
		#s = re.sub("Estimating resolution as d+", '', s)
	return s

def rotateAndSaveImage(imgPath, angle):
	rotate_command = f"convert {imgPath} -rotate {angle} -quality 100 {imgPath}"
	subprocess.run(rotate_command, shell=True, check=True)
	identify_command = f"identify -format '%w %h' {imgPath}"
	result = subprocess.run(identify_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	if result.stderr:
		print("Error in getting image dimensions:", result.stderr.decode())
		return None, None
	dimensions = result.stdout.decode().strip().split()
	width, height = map(int, dimensions)
	return width, height

def searchIterator(imgPath, debug = True):
	# Bruteforces each combination of rotation and language
	results = {}
	rotated = -1
	for i in range(4):
		if i == 0:
			w, h = rotateAndSaveImage(imgPath, 0)
		else:
			w, h = rotateAndSaveImage(imgPath, 90)
		if h > w and rotated == -1:
				rotated = i
		for lang in ['ru', 'en', 'both']:
			text = getDetectedText(imgPath, lang)
			special_symbols_removed, characters_removed, filtered_text = filterText(text, lang)
			levenshtein_distance, num_words, corrected_text = spellCheck(filtered_text)
			results[(lang, i)] = [levenshtein_distance, filtered_text, special_symbols_removed, characters_removed, num_words, corrected_text]
	rotateAndSaveImage(imgPath, 90+90*rotated) 
	minimum_distance = float('inf')
	minimum_key = None
	for key, value in results.items():
		if value[4] != 0:
			metric = value[0]/value[4]
			if metric < minimum_distance:
				minimum_distance = metric
				minimum_key = key
	if debug:
		debugInfo(minimum_key, results)
		imgName = os.path.basename(imgPath)
		createJsonFile(results, minimum_key, imgName, rotated)
	return minimum_key, results		   

def filterText(text, lang):
	if lang == 'en':
		filtered_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
	elif lang == 'ru':
		filtered_text = re.sub(r'[^а-яА-ЯёЁ0-9\s]', '', text)
	else:
		filtered_text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s]', '', text)

	# Calculate removed characters
	special_symbols_removed = len(re.findall(r'[^\w\s]', text))
	characters_removed = len(text) - len(filtered_text)

	return special_symbols_removed, characters_removed, filtered_text
	

def detect_language(word):
	if re.search('[\u0400-\u04FF]', word):
		return 'ru'
	else:
		return 'en'


def spellCheck(text):
	words = text.split()
	total_levenshtein_distance = 0
	num_words = len(words)
	corrected_words = []
	
	for word in words:
		lang = detect_language(word)
		nlp = nlp_en if lang == 'en' else nlp_ru
		spell = spell_en if lang == 'en' else spell_ru
		
		doc = nlp(word)
		for token in doc:
			if token.pos_ == 'PROPN' or spell.unknown([token.text]) == set():
				corrected_word = token.text
			else:
				corrected_word = spell.correction(token.text)
				if corrected_word != token.text:
					total_levenshtein_distance += nltk.edit_distance(token.text, corrected_word)
			corrected_words.append(corrected_word)
	corrected_text_str = ' '.join(corrected_words)
	return total_levenshtein_distance, num_words, corrected_text_str


def debugInfo(minimum_key, results):
	print("--------------------------")
	for key, contents in results.items():
		lang, orientation = key
		orientation = str(orientation)
		lang_dict = {'ru': 'RUSSIAN', 'en': 'ENGLISH', 'both': 'BOTH'}
		orientation_dict = {'0': 'ORIGINAL', '1': 'ROTATED90', '2': 'ROTATED180', '3': 'ROTATED270'}
		print(f"{orientation_dict[orientation]} image tested for {lang_dict[lang]}:")
		if contents[4] != 0:
			metric = contents[0]/contents[4]
			if key == minimum_key:
				print("Metric:", contents[0]/contents[4], "(MIN)")
			else:
				print("Metric:", contents[0]/contents[4])
		else:
			print("Metric: NO DATA (DIVISION BY ZERO)")
		print("Total LEVENSHTEIN DISTANCE:", contents[0])
		print("Number of SPECIAL SYMBOLS removed:", contents[2])
		print("Number of CHARACTERS removed:", contents[3])
		print("Total number of WORDS:", contents[4])
		print("--------------------------")

def createJsonFile(results, minimum_key, image_name, was_rotated):
	data_to_save = {
		"minimum_key": minimum_key,
		"was_rotated": was_rotated,
		"levenshtein_distance": results[minimum_key][0],
		"special_symbols_removed": results[minimum_key][2],
		"characters_removed": results[minimum_key][3],
		"num_words": results[minimum_key][4],
		"detected_text": results[minimum_key][1],
		"corrected_text": results[minimum_key][5]
	}

	base_name = os.path.basename(image_name)
	name_without_extension, _ = os.path.splitext(base_name)
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
	searchIterator(args.imgPath)
	
if __name__ == "__main__":
	main()
