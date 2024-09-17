import torch
import torch.nn.functional as F
from pylanguagetool import api
from transformers import AutoTokenizer, LongformerForMaskedLM, AutoModelForCausalLM
from langdetect import detect
import language_tool_python
import time

# Инициализация моделей и токенизаторов для русского и английского языков
ru_model_name = 'sberbank-ai/rugpt3small_based_on_gpt2'  # Модель для русского языка
en_model_name = 'gpt2'  # Модель для английского языка

# Загрузка токенизаторов и моделей
ru_tokenizer = AutoTokenizer.from_pretrained(ru_model_name)
ru_model = AutoModelForCausalLM.from_pretrained(ru_model_name)

en_tokenizer = AutoTokenizer.from_pretrained(en_model_name)
en_model = AutoModelForCausalLM.from_pretrained(en_model_name)

def load_english_lexicon():
    """Загружаем словарь английского языка из файла english.txt"""
    with open('english.txt', 'r', encoding='utf-8') as f:
        english_words = set(line.strip().lower() for line in f)
    return english_words

# Функция для загрузки русского словаря
def load_russian_lexicon():
    """Загружаем словарь русского языка из файла russian_words.txt"""
    with open('russian.txt', 'r', encoding='utf-8') as f:
        russian_words = set(line.strip().lower() for line in f)
    return russian_words

# Функция для определения языка текста
def detect_language(text):
    start_time = time.strftime("%H:%M:%S")
    print(f"Function detect_language started at: {start_time}")
    
    lang = detect(text)
    
    end_time = time.strftime("%H:%M:%S")
    print(f"Function detect_language ended at: {end_time}")
    return lang

# Функция для расчёта энтропии с использованием соответствующей модели
def calculate_entropy(text, model, tokenizer):
    start_time = time.strftime("%H:%M:%S")
    print(f"Function calculate_entropy started at: {start_time}")
    
    tokens = tokenizer(text, return_tensors='pt')
    input_ids = tokens.input_ids
    
    total_entropy = 0
    count = 0
    
    for i in range(1, input_ids.size(1) - 1):  # Исключаем специальные токены
        input_ids_masked = input_ids.clone()
        input_ids_masked[0, i] = tokenizer.mask_token_id
        
        with torch.no_grad():
            outputs = model(input_ids_masked)
            logits = outputs.logits
            probs = F.softmax(logits[0, i], dim=-1)
            log_probs = F.log_softmax(logits[0, i], dim=-1)
            entropy = -torch.sum(probs * log_probs).item()
            total_entropy += entropy
            count += 1
    
    avg_entropy = total_entropy / count if count > 0 else 0
    
    end_time = time.strftime("%H:%M:%S")
    print(f"Function calculate_entropy ended at: {end_time}")
    return avg_entropy

def language_model_score(text, model, tokenizer):
    start_time = time.strftime("%H:%M:%S")
    print(f"Function language_model_score started at: {start_time}")
    max_length = 512
    tokens = tokenizer(text, return_tensors='pt', truncation=False)  # Токенизация без обрезки
    
    total_loss = 0
    count = 0

    for i in range(0, tokens['input_ids'].size(1), max_length):
        end_index = min(i + max_length, tokens['input_ids'].size(1))
        chunk = {
            'input_ids': tokens['input_ids'][:, i:end_index],
            'attention_mask': tokens['attention_mask'][:, i:end_index]
        }

        if chunk['input_ids'].size(1) == 0:
            continue  # Пропускаем пустые куски
        
        with torch.no_grad():
            outputs = model(**chunk, labels=chunk['input_ids'])
            loss = outputs.loss
            total_loss += loss.item()
            count += 1

    if count == 0:
        print("Error: No valid chunks processed")
        return float('inf')  # Возвращаем бесконечность, если не удалось обработать ни один кусок

    perplexity = torch.exp(torch.tensor(total_loss / count)).item()
    end_time = time.strftime("%H:%M:%S")
    print(f"Function language_model_score ended at: {end_time}")
    return perplexity

# Функция для расчёта оценки качества текста (composite score)
def composite_score(text, language):
    start_time = time.strftime("%H:%M:%S")
    print(f"Function composite_score started at: {start_time}")
    weights = {'recognized': 1, 'perplexity': 1, 'errors': 1, 'invalid_ratio': 1}
    
    # Выбираем соответствующую модель и токенизатор
    if language == 'ru':
        model = ru_model
        tokenizer = ru_tokenizer
    elif language == 'en':
        model = en_model
        tokenizer = en_tokenizer
    else:
        raise ValueError(f"Unsupported language detected: {language}")
    
    # Рассчитываем метрики
    word_count = word_count_score(text)
    perplexity = language_model_score(text, model, tokenizer)
    errors = grammatical_errors_score(text, language)
    invalid_ratio = invalid_word_ratio(text)
    
    # Финальный скор
    score = (weights['recognized'] * word_count * (100 / (weights['perplexity'] * perplexity)) -  
             weights['errors'] * errors * weights['invalid_ratio'] * invalid_ratio)
    
    end_time = time.strftime("%H:%M:%S")
    print(f"Function composite_score ended at: {end_time}")
    return score, word_count, perplexity, errors, invalid_ratio

# Функция для расчёта количества грамматических ошибок
def grammatical_errors_score(text, lang):
    start_time = time.strftime("%H:%M:%S")
    print(f"Function grammatical_errors_score started at: {start_time}")
    
    if lang == 'en':
        lang_tool_lang = 'en-US'
    elif lang == 'ru':
        lang_tool_lang = 'ru-RU'
    else:
        raise ValueError(f"Unsupported language: {lang}")
    
    words = text.split()
    total_errors = 0
    chunk_size = 20  # Например, 20 слов за один раз
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        matches = api.check(chunk, api_url='https://languagetool.org/api/v2/', lang=lang_tool_lang)
        total_errors += len(matches['matches'])
    
    end_time = time.strftime("%H:%M:%S")
    print(f"Function grammatical_errors_score ended at: {end_time}")
    return total_errors


# Функция для расчёта количества слов
def word_count_score(text):
    start_time = time.strftime("%H:%M:%S")
    print(f"Function word_count_score started at: {start_time}")
    
    word_count = len(text.split())
    
    end_time = time.strftime("%H:%M:%S")
    print(f"Function word_count_score ended at: {end_time}")
    return word_count

# Функция для определения отношения несуществующих слов
def invalid_word_ratio(text):
    start_time = time.strftime("%H:%M:%S")
    print(f"Function invalid_word_ratio started at: {start_time}")
    
    english_words = load_english_lexicon()  # Загрузка словаря английского языка
    russian_words = load_russian_lexicon()  # Загрузка словаря русского языка
    
    tokens = text.split()
    total_words = len(tokens)
    invalid_words = [word for word in tokens if word.lower() not in english_words and word.lower() not in russian_words]
    ratio = len(invalid_words) / total_words if total_words > 0 else 0
    
    end_time = time.strftime("%H:%M:%S")
    print(f"Function invalid_word_ratio ended at: {end_time}")
    return ratio


