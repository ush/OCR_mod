import pytesseract
from PIL import Image
import easyocr
import pyocr
import pyocr.builders
import numpy as np
import nltk
import re
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

parser = argparse.ArgumentParser(description='testing OCR and pytesseract')
parser.add_argument('--img_path', type=str, required=True, help='Path to the image file')
parser.add_argument('--lang', type=str, default='ru', help='Language for OCR (default: ru)')
args = parser.parse_args()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('udhr')
russian_stopwords = set(stopwords.words('russian'))

# OCR Function
def read_text_from_image(image_path, ocr_tool='pytesseract', lang='rus'):
    text = ""

    if ocr_tool == 'pytesseract':
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=lang)

    elif ocr_tool == 'easyocr':
        reader = easyocr.Reader([lang])
        result = reader.readtext(image_path)
        text = "\n".join([detection[1] for detection in result])

    elif ocr_tool == 'pyocr':
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            raise RuntimeError("No OCR tool found")
        tool = tools[0]
        img = Image.open(image_path)
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
    # Load the pre-trained GPT-2 model and tokenizer
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Encode the input text
    inputs = tokenizer(text, return_tensors='pt')
    
    # Compute the loss
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
    
    # Compute the perplexity
    perplexity = torch.exp(loss).item()
    return perplexity
    

def composite_score(text):
    weights = {'recognized': 1, 'perplexity': 1, 'errors': 1}
    word_count = word_count_score(text)
    perplexity = language_model_score(text)
    errors = grammatical_errors_score(text)
    
    # Combine scores with weights
    score = (weights['recognized'] * word_count - 
             weights['perplexity'] * perplexity - 
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

with open('pytesseract.txt', 'w') as f:
    f.write(texts['pytesseract'])
    
with open('easyocr.txt', 'w') as f:
    f.write(texts['easyocr'])
    
with open('pyocr.txt', 'w') as f:
    f.write(texts['pyocr'])

# Сравниваем результаты по метрике
results = compare_texts(texts['pytesseract'], texts['easyocr'], texts['pyocr'])
print_results(results)

