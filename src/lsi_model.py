

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from typing import List, Dict, Tuple


class LSIModel:

    
    def __init__(self, n_topics=10, min_df=1, max_df=0.9):

        self.n_topics = n_topics
        self.min_df = min_df
        self.max_df = max_df
        
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            lowercase=False,  # Already cleaned
            token_pattern=r'(?u)\b\w+\b'
        )
        
        self.svd = TruncatedSVD(
            n_components=n_topics,
            random_state=42
        )
        
        self.normalizer = Normalizer(copy=False)
        
        self.tfidf_matrix = None
        self.lsi_matrix = None
        self.vocabulary = None
    
    def fit(self, processed_docs: List[Dict]) -> 'LSIModel':

        texts = [doc['processed_text'] for doc in processed_docs]
        
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        self.vocabulary = self.vectorizer.get_feature_names_out()
        
        self.lsi_matrix = self.svd.fit_transform(self.tfidf_matrix)
        
        self.lsi_matrix = self.normalizer.fit_transform(self.lsi_matrix)
        
        print(f"LSI Model fitted:")
        print(f"  - Documents: {len(processed_docs)}")
        print(f"  - Vocabulary size: {len(self.vocabulary)}")
        print(f"  - Topics: {self.n_topics}")
        print(f"  - Explained variance: {self.svd.explained_variance_ratio_.sum():.2%}")
        
        return self
    
    def transform(self, processed_docs: List[Dict]) -> np.ndarray:

        texts = [doc['processed_text'] for doc in processed_docs]
        
        tfidf = self.vectorizer.transform(texts)
        
        lsi = self.svd.transform(tfidf)
        
        lsi = self.normalizer.transform(lsi)
        
        return lsi
    
    def fit_transform(self, processed_docs: List[Dict]) -> np.ndarray:

        self.fit(processed_docs)
        return self.lsi_matrix
    
    def get_top_terms_per_topic(self, n_terms=10) -> Dict[int, List[Tuple[str, float]]]:

        if self.svd.components_ is None:
            raise ValueError("Model not fitted yet")
        
        topic_terms = {}
        
        for topic_idx in range(self.n_topics):
            component = self.svd.components_[topic_idx]
            
            top_indices = np.argsort(np.abs(component))[::-1][:n_terms]
            
            terms = [(self.vocabulary[i], component[i]) for i in top_indices]
            
            topic_terms[topic_idx] = terms
        
        return topic_terms
    
    def print_topics(self, n_terms=10):

        topics = self.get_top_terms_per_topic(n_terms)
        
        print("\n=== LSI Topics ===")
        for topic_idx, terms in topics.items():
            print(f"\nTopic {topic_idx}:")
            term_str = ", ".join([f"{term}({weight:.3f})" for term, weight in terms])
            print(f"  {term_str}")
    
    def get_document_topic_distribution(self, doc_idx: int) -> np.ndarray:

        if self.lsi_matrix is None:
            raise ValueError("Model not fitted yet")
        
        return self.lsi_matrix[doc_idx]
    
    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:

        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)


if __name__ == "__main__":
    from preprocessing import DocumentPreprocessor
    
    sample_docs = [
        {'id': 1, 'title': 'Football', 'content': 'Football match victory'},
        {'id': 2, 'title': 'Soccer', 'content': 'Soccer game highlights'},
        {'id': 3, 'title': 'Election', 'content': 'Election results announced'}
    ]
    
    preprocessor = DocumentPreprocessor()
    processed = preprocessor.preprocess_documents(sample_docs)
    
    lsi = LSIModel(n_topics=2)
    lsi_vectors = lsi.fit_transform(processed)
    
    print(f"\nLSI Vectors shape: {lsi_vectors.shape}")
    lsi.print_topics(n_terms=5)

