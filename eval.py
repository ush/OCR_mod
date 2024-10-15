import torch
import numpy as np
import torch.nn.functional as F
from pylanguagetool import api
from transformers import AutoTokenizer, LongformerForMaskedLM, AutoModelForCausalLM
from langdetect import detect
import language_tool_python
import time

ru_model_name = 'sberbank-ai/rugpt3small_based_on_gpt2'
en_model_name = 'gpt2'

ru_tokenizer = AutoTokenizer.from_pretrained(ru_model_name)
ru_model = AutoModelForCausalLM.from_pretrained(ru_model_name)

en_tokenizer = AutoTokenizer.from_pretrained(en_model_name)
en_model = AutoModelForCausalLM.from_pretrained(en_model_name)

def load_english_lexicon():
    """Загружаем словарь английского языка из файла english.txt"""
    with open('english.txt', 'r', encoding='utf-8') as f:
        english_words = set(line.strip().lower() for line in f)
    return english_words

def load_russian_lexicon():
    """Загружаем словарь русского языка из файла russian_words.txt"""
    with open('russian.txt', 'r', encoding='utf-8') as f:
        russian_words = set(line.strip().lower() for line in f)
    return russian_words

def detect_language(text):
    start_time = time.strftime("%H:%M:%S")
    print(f"Function detect_language started at: {start_time}")
    
    lang = detect(text)
    
    end_time = time.strftime("%H:%M:%S")
    print(f"Function detect_language ended at: {end_time}")
    return lang

def language_model_score(text, language):
    start_time = time.strftime("%H:%M:%S")
    print(f"Function language_model_score started at: {start_time}")
   
    if language == 'ru':
        model = ru_model
        tokenizer = ru_tokenizer
    elif language == 'en':
        model = en_model
        tokenizer = en_tokenizer
    else:
        raise ValueError(f"Unsupported language detected: {language}")
    
    max_length = 512
    tokens = tokenizer(text, return_tensors='pt', truncation=False)
    
    total_loss = 0
    count = 0

    for i in range(0, tokens['input_ids'].size(1), max_length):
        end_index = min(i + max_length, tokens['input_ids'].size(1))
        chunk = {
            'input_ids': tokens['input_ids'][:, i:end_index],
            'attention_mask': tokens['attention_mask'][:, i:end_index]
        }

        if chunk['input_ids'].size(1) == 0:
            continue
        
        with torch.no_grad():
            outputs = model(**chunk, labels=chunk['input_ids'])
            loss = outputs.loss
            total_loss += loss.item()
            count += 1

    if count == 0:
        print("Error: No valid chunks processed")
        return float('inf')

    perplexity = torch.exp(torch.tensor(total_loss / count)).item()
    end_time = time.strftime("%H:%M:%S")
    print(f"Function language_model_score ended at: {end_time}")
    return perplexity

def composite_score(text, language):
    start_time = time.strftime("%H:%M:%S")
    print(f"Function composite_score started at: {start_time}")
    
    word_count = word_count_score(text)
    perplexity = language_model_score(text, language)
    errors = grammatical_errors_score(text, language)
    invalid_ratio = invalid_word_ratio(text)
    
    score = (word_count * (1 / np.log(perplexity + 1)) -  
             errors * invalid_ratio)
    
    end_time = time.strftime("%H:%M:%S")
    print(f"Function composite_score ended at: {end_time}")
    return score, word_count, perplexity, errors, invalid_ratio

def composite_score_tech(word_count, perplexity, errors, invalid_ratio):
    start_time = time.strftime("%H:%M:%S")
    print(f"Function composite_score started at: {start_time}")
    
    score = (word_count * (1 / np.log(perplexity + 1)) -  
             errors * invalid_ratio)
    
    end_time = time.strftime("%H:%M:%S")
    print(f"Function composite_score ended at: {end_time}")
    return score

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
    chunk_size = 20
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        matches = api.check(chunk, api_url='https://languagetool.org/api/v2/', lang=lang_tool_lang)
        total_errors += len(matches['matches'])
    
    end_time = time.strftime("%H:%M:%S")
    print(f"Function grammatical_errors_score ended at: {end_time}")
    return total_errors


def word_count_score(text):
    start_time = time.strftime("%H:%M:%S")
    print(f"Function word_count_score started at: {start_time}")
    
    word_count = len(text.split())
    
    end_time = time.strftime("%H:%M:%S")
    print(f"Function word_count_score ended at: {end_time}")
    return word_count

def invalid_word_ratio(text):
    start_time = time.strftime("%H:%M:%S")
    print(f"Function invalid_word_ratio started at: {start_time}")
    
    english_words = load_english_lexicon()
    russian_words = load_russian_lexicon()
    
    tokens = text.split()
    total_words = len(tokens)
    invalid_words = [word for word in tokens if word.lower() not in english_words and word.lower() not in russian_words]
    ratio = len(invalid_words) / total_words if total_words > 0 else 0
    
    end_time = time.strftime("%H:%M:%S")
    print(f"Function invalid_word_ratio ended at: {end_time}")
    return ratio


