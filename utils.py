import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import torch
import pandas as pd
import json

nltk.download('wordnet')
nltk.download('punkt')
russian_lemmatizer = WordNetLemmatizer()

def preprocessing_text(text):
    text = text.lower()
    punct = string.punctuation
    text = ''.join([char for char in text if char not in punct])

    return text
'''
def embeddings_matching(query, corpus_to_match, tokenizer, model):
    corpus = corpus_to_match.copy()
    corpus.append(query)

    for idx, text in enumerate(corpus):
        corpus[idx] = preprocessing_text(text)
        
    for key in key_phrases.keys():
        if key in query:
            return key_phrases[key]
    
    encoded_corpus = tokenizer(corpus, padding=True, return_tensors="pt")
    input_ids = encoded_corpus['input_ids']
    attention_mask = encoded_corpus['attention_mask']

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
        hidden_state = output.last_hidden_state[:, 0, :]
    
    cosine_similarities = cosine_similarity(hidden_state[-1].reshape(1, -1), hidden_state[:-1])
    most_similar_index = cosine_similarities.argmax()
    most_similar_string = corpus_to_match[most_similar_index]

    return most_similar_string
'''

def key_words_matching(query, corpus_key_words):

    query_tokens = preprocessing_text(query)
    query_tokens = query_tokens.split()
    lemmatized_query = [russian_lemmatizer.lemmatize(word) for word in query_tokens]

    key_words_copy = []
    for _, list_keys in corpus_key_words.items():
        temp = []
        for key in list_keys:
            temp.append(russian_lemmatizer.lemmatize(key))
            
        key_words_copy.append(temp)

    accuracy = [0] * len(corpus_key_words)
    for idx, key_words in enumerate(key_words_copy):
        for key in key_words:
            if key in lemmatized_query:
                accuracy[idx] += 1

        accuracy[idx] /= len(key_words)

    accuracy = np.array(accuracy)
    result_idx = np.argmax(accuracy)

    return accuracy[result_idx], list(corpus_key_words.keys())[result_idx]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def embeddings_mathicng_v2(query, corpus_of_categories, tokenizer, model):

    corpus = corpus_of_categories.copy()
    corpus.append(query)

    encoded_input = tokenizer(corpus, padding=True, truncation=True, max_length=24, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    cosine_similarities = cosine_similarity(sentence_embeddings[-1].reshape(1, -1), sentence_embeddings[:-1])
    most_similar_index = cosine_similarities.argmax()
    most_similar_string = corpus[most_similar_index]

    return most_similar_string

def find_standarts(name_standard, name_query):
    standarts = pd.read_csv('standarts_list.csv') 
    result = []

    for _, row in standarts.iterrows():
        if name_standard in row['Группа продукции']:
            cur_stand = row['Обозначение и наименование документов в области стандартизации']
            result.append(cur_stand)
              
    return result

def read_corpus_47():
    with open('data.json', 'r') as json_file:
        key_word_loaded_dataset12k = json.load(json_file)

    return list(key_word_loaded_dataset12k.keys())

def read_corpus_18():
    with open('categories-18.txt', 'r', encoding='utf-8') as file:
        key_word_loaded_dataset12k = file.read()

    return key_word_loaded_dataset12k.split('\n')

def read_corpus_5():
    with open('categories-5.txt', 'r', encoding='utf-8') as file:
        key_word_loaded_dataset12k = file.read()

    return key_word_loaded_dataset12k.split('\n')

def generate_categories(name_category, corpus_47, corpus_18, corpus_5, tokenizer, model):
    #matching_47 = embeddings_mathicng_v2(name_category, corpus_47, tokenizer=tokenizer, model=model)
    matching_18 = embeddings_mathicng_v2(name_category, corpus_18, tokenizer=tokenizer, model=model)
    matching_5 = embeddings_mathicng_v2(name_category, corpus_5, tokenizer=tokenizer, model=model)

    return matching_18, matching_5 

def standarts(name_query, tokenizer, model):
    corpus_47 = read_corpus_47()
    corpus_18 = read_corpus_18()
    corpus_5 = read_corpus_5()
    match_18, match_5 = generate_categories(name_query,
        corpus_47,
        corpus_18,
        corpus_5,
        tokenizer,
        model
    )

    standarts = find_standarts(match_5, name_query)

    return standarts, match_18, match_5

def calculate_cosine_similarity(string1, string2):
  
    def preprocess_text(text):
        # Приведение к нижнему регистру
        text = text.lower()
        # Удаление специальных символов и цифр (включая точки с запятой)
        text = ''.join(char for char in text if char.isalnum() or char.isspace())
        # Удаление лишних пробелов
        text = ' '.join(text.split())

        return text

    # Предобработка двух строк
    preprocessed_strings = [preprocess_text(string1), preprocess_text(string2)]

    # Создаем векторизатор TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Преобразуем строки в числовое представление TF-IDF
    tfidf_matrix = vectorizer.fit_transform(preprocessed_strings)

    # Вычисляем косинусное сходство между двумя строками
    cosine_similarity_value = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return cosine_similarity_value[0][0]