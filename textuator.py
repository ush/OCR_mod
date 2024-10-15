import re
import os
import pyocr
import argparse
import numpy as np
import pyocr.builders
from PIL import Image
from os import listdir
from os.path import isfile, join
import itertools

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default=None)
args = parser.parse_args()

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

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

def filterText(text, lang):
	if lang == 'en':
		filtered_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
	elif lang == 'ru':
		filtered_text = re.sub(r'[^а-яА-ЯёЁ0-9\s]', '', text)

	special_symbols_removed = len(re.findall(r'[^\w\s]', text))
	characters_removed = len(text) - len(filtered_text)
	return special_symbols_removed, characters_removed, filtered_text
	
def getTextSaveToFile(filepath, path, lang, i):
    text = getDetectedText(filepath, lang)
    special_symbols_removed, characters_removed, filtered_text = filterText(text, lang)
    txt_filename = f"{i}.txt"
    save_path = os.path.join(path, txt_filename)
    with open(save_path, "w", encoding='utf-8') as f:
        f.write(filtered_text)

def progon(path):
    allfiles = []
    for f in listdir(path):
        if f.endswith(".jpg"):
            allfiles.append(join(path, f))
    
    allfiles = sorted_alphanumeric(allfiles)
    print(allfiles)
    
    i = 1
    for filepath in allfiles:
        if i <= 50:
            getTextSaveToFile(filepath, path, "en", i)
        else:
            getTextSaveToFile(filepath, path, "ru", i)
        i += 1
    return 0

def main():
    progon(args.path)
    print("Texts generated!")

if __name__ == "__main__":
	main()
