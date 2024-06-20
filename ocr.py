import pytesseract
from PIL import Image
import easyocr
import pyocr
import pyocr.builders
import numpy as np
import nltk
import re
import cv2
import io
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
from pylanguagetool import api
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import LongformerTokenizer, LongformerForMaskedLM
from skimage.filters import threshold_local

parser = argparse.ArgumentParser(description='testing OCR and pytesseract')
parser.add_argument('--img_path', type=str, required=True, help='Path to the image file')
parser.add_argument('--lang', type=str, default='ru', help='Language for OCR (default: ru)')
args = parser.parse_args()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('udhr')
russian_stopwords = set(stopwords.words('russian'))

def complex_preprocess_image(image_path, scale_factor=2):
	# Read the image in color
	image = cv2.imread(image_path)

	# Convert to grayscale for some operations but keep the original
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Апскейлинг изображения
	upscaled_gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
	
	#cv2.imwrite("./upscaled.jpg", upscaled_gray)
    	
    	# Enhance contrast
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	enhanced_gray = clahe.apply(upscaled_gray)

	#cv2.imwrite("./enhanced.jpg", enhanced_gray)
	
	# Применение уменьшения шума
	#denoised_gray = cv2.fastNlMeansDenoising(enhanced_gray, None, 30, 7, 21)
	
	#cv2.imwrite("./denoised.jpg", denoised_gray)
	
	# Применение сглаживания для уменьшения изрезанности
	#blurred = cv2.GaussianBlur(denoised_gray, (5, 5), 0)
	
	#cv2.imwrite("./blurred.jpg", blurred)

	# Применение адаптивного порогового значения для создания бинарного изображения
	#_, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	
	#cv2.imwrite("./binary.jpg", binary_image)

	# Deskewing
	#coords = np.column_stack(np.where(binary_image > 0))
	#(h, w) = image.shape[:2]
	#center = (w // 2, h // 2)
	#M = cv2.getRotationMatrix2D(center, 0, 1.0)
	#deskewed_image = cv2.warpAffine(binary_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	
	#cv2.imwrite("./final.jpg", deskewed_image)

	return enhanced_gray
	
def simple_preprocess_image(image_path, name):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.equalizeHist(image)
    image = cv2.medianBlur(image, 3)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("./" + name + ".jpg", image)
    return image

# OCR Function
def read_text_from_image(image_path, ocr_tool='pytesseract', lang='rus'):
	text = ""

	if ocr_tool == 'pytesseract':
		desk_img = complex_preprocess_image(image_path)
		img = Image.fromarray(desk_img)
		text = pytesseract.image_to_string(img, lang=lang)

	elif ocr_tool == 'easyocr':
		reader = easyocr.Reader([lang])
		desk_img = complex_preprocess_image(image_path)
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
		desk_img = complex_preprocess_image(image_path)
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
	
# Function to calculate grammatical errors score using LanguageTool
def grammatical_errors_score(text):
	matches = api.check(text, api_url='https://languagetool.org/api/v2/', lang = 'ru-RU')
	return len(matches['matches'])
	
def word_count_score(text):
	return len(text.split(' '))
	
def language_model_score(text):
	model_name = 'allenai/longformer-base-4096'
	model = LongformerForMaskedLM.from_pretrained(model_name)
	tokenizer = LongformerTokenizer.from_pretrained(model_name)
	
	# Токенизация текста
	tokens = tokenizer.encode(text, return_tensors='pt')
	max_length = 4096
	num_chunks = tokens.size(1) // max_length + 1
	
	total_loss = 0
	count = 0
	
	for i in range(num_chunks):
		start = i * max_length
		end = min((i + 1) * max_length, tokens.size(1))
		chunk = tokens[:, start:end]
		
		# Создание входных данных для модели
		inputs = tokenizer.decode(chunk[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
		inputs = tokenizer(inputs, return_tensors='pt')
		
		# Предсказание вероятностей и вычисление loss
		with torch.no_grad():
			outputs = model(**inputs, labels=inputs['input_ids'])
			loss = outputs.loss
		total_loss += loss.item()
		count += 1
	
	# Вычисление perplexity
	perplexity = torch.exp(torch.tensor(total_loss / count)).item()
	return perplexity
	

def composite_score(text):
	weights = {'recognized': 1, 'perplexity': 1, 'errors': 1}
	word_count = word_count_score(text)
	perplexity = language_model_score(text)
	errors = grammatical_errors_score(text)
	
	#print("----------------------------")
	#print("word count ", word_count)
	#print("perplexity ",perplexity)
	#print("errors ",errors)
	#print("----------------------------")
	
	# Combine scores with weights
	score = (weights['recognized'] * word_count * (1 / perplexity) -  
			 weights['errors'] * errors)
	
	return score


# Comprehensive function for comparison
def compare_texts(text1, text2, text3):
	texts = [preprocess_text(text1), preprocess_text(text2), preprocess_text(text3)]
	results = {}
	
	# Grammatical errors score
	results['Grammatical Errors'] = [grammatical_errors_score(text) for text in texts]
	
	results['Language Model Perplexity'] = [language_model_score(text) for text in texts]
	
	results['Word Count'] = [word_count_score(text) for text in texts]
	
	results['Composite Score'] = [composite_score(text) for text in texts]
	
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
		texts.update({tool: preprocess_text(read_text_from_image(image_path, tool, language))})

	# Выводим распознанный текст каждого OCR-инструмента
	#for tool, text in texts.items():
	#    print(f"\nРаспознанный текст ({tool}):\n{text}\n")

	# Сравниваем результаты по метрике
	results = compare_texts(texts['pytesseract'], texts['easyocr'], texts['pyocr'])
	composite = results['Composite Score']
	word = results['Word Count']
	perplexity = results['Language Model Perplexity']
	errors = results['Grammatical Errors']

	with open('pytesseract.txt', 'a') as f:
			f.write(str(composite[0]) + "," + str(word[0]) + "," + str(perplexity[0]) + "," + str(errors[0]) + "\n")
	#    	f.write("\n" + texts['pytesseract'] + "\n")
	
	with open('easyocr.txt', 'a') as f:
			f.write(str(composite[1]) + "," + str(word[1]) + "," + str(perplexity[1]) + "," + str(errors[1]) + "\n")
	#    	f.write("\n" + texts['easyocr'] + "\n")
	
	with open('pyocr.txt', 'a') as f:
	    	f.write(str(composite[2]) + "," + str(word[2]) + "," + str(perplexity[2]) + "," + str(errors[2]) + "\n")
	#    	f.write("\n" + texts['pyocr'] + "\n")


if __name__ == "__main__":
    main()
