from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List
from nltk.stem.porter import PorterStemmer
import re

STOPWORDS = set(stopwords.words('english'))

def remove_special_characters(text: str) -> str:
    return re.sub(r'[^a-zA-Z\s]', '', text)

def remove_extra_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def remove_mentions_hashtags(text: str) -> str:
    return re.sub(r'@\w+|#\w+', '', text)

def remove_urls(text: str) -> str:
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def tokenize_text(text: str) -> List[str]:    
    return word_tokenize(text.lower())

def remove_stop_words(tokens: List[str]) -> List[str]:
    return [token for token in tokens if token.lower() not in STOPWORDS]

def stem_text(tokens: List[str]) -> List[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def preprocess(text: str) -> str:
    """
    Preprocess the input text by removing URLs, mentions, hashtags, special characters, extra spaces,
    tokenizing, removing stop words, and do Porter stemming.
    
    Args:
        text (str): The input text.
        
    Returns:
        str: The preprocessed text.
    """
    text = remove_urls(text) #Remove URLs from the input text.
    text = remove_mentions_hashtags(text) #Remove mentions and hashtags from the input text.
    text = remove_special_characters(text) # Remove special characters from the input text.
    text = remove_extra_spaces(text) # Remove extra spaces from the input text.
    tokens = tokenize_text(text) #Convert the input text to lowercase and tokenize it.
    tokens = remove_stop_words(tokens) #Remove stop words from the input list of tokens.
    tokens = stem_text(tokens) #Stem the input list of tokens using the PorterStemmer.
    return ' '.join(tokens)

