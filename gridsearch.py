import re
import pyocr
import numpy as np
import pyocr.builders
from PIL import Image
from os import listdir
from os.path import isfile, join
from eval import composite_score
import itertools

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

def progon():
    path = './Shitted/'
    allfiles = []
    for f in listdir(path):
        allfiles.append(join(path, f))
    
    params_rec = [0.25]
    params_perp = [2]
    params_err = [1.75]
    params_inv = [1]
    
    params_comb = list(itertools.product(params_rec, params_perp, params_err, params_inv))
    
    best_params = {}
    i = 0
    for rec, perp, err, inv in params_comb:
        i += 1
        results = []

        for filepath in allfiles:
            print(filepath)
            max_res = 0
            text = getDetectedText(filepath, "ru")
            print(text)
            special_symbols_removed, characters_removed, filtered_text = filterText(text, "ru")
            composite_score_value_ru, word_count, perplexity, errors, invalid_ratio = composite_score(filtered_text, "ru", rec, perp, err, inv)
            text = getDetectedText(filepath, "en")
            print(text)
            special_symbols_removed, characters_removed, filtered_text = filterText(text, "en")
            composite_score_value_en, word_count, perplexity, errors, invalid_ratio = composite_score(filtered_text, "en", rec, perp, err, inv)
            results.append(max(composite_score_value_ru, composite_score_value_en))
    
        pyocr_mean = np.mean(results)
        pyocr_std = np.std(results)
        res_min = min(results)
        res_max = max(results)

        Z_90 = 1.645
        e_max = 0.1 * pyocr_mean
        n_needed = (Z_90 * pyocr_std / e_max) ** 2
        best_params[i] = [n_needed, pyocr_mean, pyocr_std, res_min, res_max, rec, perp, err, inv]
    
    ind = min(best_params, key=lambda k: best_params[k][0])
    return best_params[ind]

def main():
    best_params = progon()
    print("Best paramters: ")
    print("N_needed - ", best_params[0])
    print("Mean - ", best_params[1])
    print("Std - ", best_params[2])
    print("Min - ", best_params[3])
    print("Max - ", best_params[4])
    print("rec parameter - ", best_params[5])
    print("perp pararmeter - ", best_params[6])
    print("err parameter - ", best_params[7])
    print("inv parameter - ", best_params[8])

if __name__ == "__main__":
	main()
