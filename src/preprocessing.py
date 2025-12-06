

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import List, Dict


class DocumentPreprocessor:

    
    def __init__(self, use_stemming=True, remove_stopwords=True):

        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.stemmer = PorterStemmer()
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        
        try:
            word_tokenize("test")
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def clean_text(self, text: str) -> str:

        text = text.lower()
        
        text = re.sub(r'http\S+|www\S+', '', text)
        
        text = re.sub(r'\S+@\S+', '', text)
        
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:

        return word_tokenize(text)
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:

        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:

        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess_text(self, text: str) -> List[str]:

        cleaned = self.clean_text(text)
        
        tokens = self.tokenize(cleaned)
        
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        
        if self.use_stemming:
            tokens = self.stem_tokens(tokens)
        
        tokens = [token for token in tokens if len(token) >= 2]
        
        return tokens
    
    def preprocess_documents(self, documents: List[Dict[str, str]]) -> List[Dict]:

        processed_docs = []
        
        for doc in documents:
            full_text = f"{doc.get('title', '')} {doc.get('content', '')}"
            
            tokens = self.preprocess_text(full_text)
            
            processed_doc = {
                'id': doc.get('id'),
                'title': doc.get('title', ''),
                'original_content': doc.get('content', ''),
                'tokens': tokens,
                'processed_text': ' '.join(tokens)
            }
            if 'category' in doc:
                processed_doc['category'] = doc['category']
            if 'source' in doc:
                processed_doc['source'] = doc['source']
            if 'url' in doc:
                processed_doc['url'] = doc['url']
            processed_docs.append(processed_doc)
        
        return processed_docs
    
    def get_vocabulary(self, processed_docs: List[Dict]) -> set:

        vocab = set()
        for doc in processed_docs:
            vocab.update(doc['tokens'])
        return vocab


if __name__ == "__main__":
    sample_docs = [
        {
            'id': 1,
            'title': 'Football Match',
            'content': 'The football match ended in a spectacular victory for the home team.'
        },
        {
            'id': 2,
            'title': 'Election News',
            'content': 'Election results were announced today with surprising outcomes.'
        }
    ]
    
    preprocessor = DocumentPreprocessor(use_stemming=True, remove_stopwords=True)
    processed = preprocessor.preprocess_documents(sample_docs)

