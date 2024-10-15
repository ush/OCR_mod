import argparse
from pathlib import Path
from PIL import Image
import pytesseract
import pyocr
import pyocr.builders
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
import os
from eval import grammatical_errors_score, word_count_score, language_model_score, composite_score, invalid_word_ratio
from grayscale import complex_preprocess_image, no_preprocess_image

# Загрузка языковых данных для NLTK
import nltk
nltk.download('punkt')
nltk.download('stopwords')

os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract/tessdata/'

parser = argparse.ArgumentParser(description='OCR и обработка текста с изображений.')
parser.add_argument('--img_path', type=str, required=True, help='Путь к изображению.')
parser.add_argument('--out_path', type=str, required=True, help='Путь к изображению.')
args = parser.parse_args()

# Поддержка языков
english_stopwords = stopwords.words('english')
russian_stopwords = stopwords.words('russian')

# Чтение текста с изображения
def read_text_from_image(img_path, ocr_tool='pyocr', lang='rus'):
    img = Image.open(img_path)
    text = pytesseract.image_to_string(img, lang=lang)
    return text

# Предобработка текста
def preprocess_text(text, lang='rus'):
    text = text.lower()
    
    # Удаление символов, не относящихся к буквам и пробелам
    if lang == 'rus':
        text = re.sub(r'[^А-Яа-я\s]', '', text)
    elif lang == 'eng':
        text = re.sub(r'[^A-Za-z\s]', '', text)
    
    text = text.replace("\n", " ")  # Удаление новых строк
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    
    tokens = []
    if lang == 'rus':
        tokens = [word for word in word_tokenize(text, language='russian') if word not in russian_stopwords]
    elif lang == 'eng':
        tokens = [word for word in word_tokenize(text, language='english') if word not in english_stopwords]
    
    return ' '.join(tokens)

# Вывод результатов
def print_metrics(results):
    print("Метрики текста:")
    for metric, value in results.items():
        print(f"{metric}: {value}")

# Определение языка на основе метрики
def process_image(img_path):
    languages = ['rus', 'eng']  # Список языков
    best_score = None
    best_text = None
    best_lang = None
    best_details = None

    for lang in languages:
        text = read_text_from_image(img_path, 'pyocr', lang)
        processed_text = preprocess_text(text, lang)
        
        eval_lang = ''
        if lang == 'rus':
            eval_lang = 'ru'
        elif lang == 'eng':
            eval_lang = 'en'
        # Применение метрик
        score, word_count, perplexity, errors, invalid_ratio = composite_score(processed_text, eval_lang)

        # Сравнение результатов для обоих языков, оставляем лучший
        if best_score is None or score > best_score:
            best_score = score
            best_text = processed_text
            best_lang = lang
            best_details = {
                "Метрика": score,
                "Количество слов": word_count,
                "Perplexity": perplexity,
                "Ошибки": errors,
                "Отношение неверных слов": invalid_ratio
            }
        return best_details

def main():
    res = process_image(args.img_path)
    with open(args.out_path, 'w') as f:
        f.write(str(res))
        
# Консольный интерфейс
if __name__ == "__main__":
    main()

