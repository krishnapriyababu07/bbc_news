import requests

def fetch_news_article(api_key):
    url = ('https://newsapi.org/v2/top-headlines?'
           'country=us&'
           'apiKey={}'.format(api_key))
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'ok':
        # Take the first article for simplicity
        article = data['articles'][0]
        title = article['title']
        content = article['content']
        return title, content
    else:
        raise Exception('Failed to fetch news articles')

api_key = 'f29e3591bd0f4a6f815d0a110131649a'
title, content = fetch_news_article(api_key)
print("Title:", title)
print("Content:", content)
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def extract_entities_nltk(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    tree = ne_chunk(pos_tags, binary=False)
    named_entities = []
    
    for subtree in tree:
        if isinstance(subtree, nltk.Tree):
            entity_name = " ".join([word for word, pos in subtree.leaves()])
            entity_type = subtree.label()
            named_entities.append((entity_name, entity_type))
    
    return named_entities

entities_nltk = extract_entities_nltk(content)
print("Entities extracted by NLTK:")
for entity in entities_nltk:
    print(entity)

import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_entities_spacy(text):
    doc = nlp(text)
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]
    return named_entities

entities_spacy = extract_entities_spacy(content)
print("Entities extracted by spaCy:")
for entity in entities_spacy:
    print(entity)


def compare_entities(nltk_entities, spacy_entities):
    nltk_set = set(nltk_entities)
    spacy_set = set(spacy_entities)

    print("\nEntities found by both NLTK and spaCy:")
    for entity in nltk_set & spacy_set:
        print(entity)

    print("\nEntities found only by NLTK:")
    for entity in nltk_set - spacy_set:
        print(entity)

    print("\nEntities found only by spaCy:")
    for entity in spacy_set - nltk_set:
        print(entity)

compare_entities(entities_nltk, entities_spacy)
