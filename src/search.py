

import numpy as np
from typing import List, Dict, Tuple
from src.preprocessing import DocumentPreprocessor
from src.lsi_model import LSIModel


class DocumentSearcher:

    
    def __init__(self, preprocessor: DocumentPreprocessor, lsi_model: LSIModel):

        self.preprocessor = preprocessor
        self.lsi_model = lsi_model
        
        self.documents = []
        self.vectors = None
        self.labels = None
    
    def index_documents(self, processed_docs: List[Dict], 
                       vectors: np.ndarray, 
                       labels: np.ndarray = None):

        self.documents = processed_docs
        self.vectors = vectors
        self.labels = labels
        
        print(f"Indexed {len(processed_docs)} documents for searching")
    
    def search(self, query: str, top_k=5, cluster_filter=None, category_filter=None) -> List[Dict]:

        query_doc = {'id': -1, 'title': '', 'content': query}
        processed_query = self.preprocessor.preprocess_documents([query_doc])[0]
        
        query_vector = self.lsi_model.transform([processed_query])[0]
        
        similarities = self._compute_similarities(query_vector)
        
        if cluster_filter is not None and self.labels is not None:
            cluster_mask = self.labels == cluster_filter
            similarities = similarities * cluster_mask
        
        if category_filter is not None:
            category_mask = np.array([
                doc.get('category', '').lower() == category_filter.lower() 
                for doc in self.documents
            ])
            similarities = similarities * category_mask
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            result = {
                'rank': len(results) + 1,
                'document': self.documents[idx],
                'score': float(similarities[idx]),
                'cluster': int(self.labels[idx]) if self.labels is not None else None
            }
            results.append(result)
        
        return results
    
    def search_by_cluster(self, cluster_id: int) -> List[Dict]:

        if self.labels is None:
            raise ValueError("No cluster labels available")
        
        cluster_mask = self.labels == cluster_id
        cluster_docs = [doc for i, doc in enumerate(self.documents) if cluster_mask[i]]
        
        return cluster_docs
    
    def _compute_similarities(self, query_vector: np.ndarray) -> np.ndarray:

        similarities = np.dot(self.vectors, query_vector)
        
        
        return similarities
    
    def print_results(self, results: List[Dict], show_content=False):

        if not results:
            print("No results found.")
            return
        
        print(f"\n=== Search Results ({len(results)} documents) ===\n")
        
        for result in results:
            doc = result['document']
            print(f"Rank {result['rank']}: {doc['title']} (Score: {result['score']:.3f})")
            
            if result['cluster'] is not None:
                print(f"  Cluster: {result['cluster']}")
            
            if show_content:
                content = doc.get('original_content', '')[:200]
                print(f"  Content: {content}...")
            
            print()


if __name__ == "__main__":
    from preprocessing import DocumentPreprocessor
    from lsi_model import LSIModel
    from clustering import HierarchicalClustering
    
    docs = [
        {'id': 1, 'title': 'Football Match', 'content': 'Football match ended in victory'},
        {'id': 2, 'title': 'Soccer Game', 'content': 'Soccer game highlights'},
        {'id': 3, 'title': 'Election', 'content': 'Election results announced'}
    ]
    
    preprocessor = DocumentPreprocessor()
    processed = preprocessor.preprocess_documents(docs)
    
    lsi = LSIModel(n_topics=2)
    vectors = lsi.fit_transform(processed)
    
    clustering = HierarchicalClustering(n_clusters=2)
    labels = clustering.fit_predict(vectors)
    
    searcher = DocumentSearcher(preprocessor, lsi)
    searcher.index_documents(processed, vectors, labels)
    
    results = searcher.search("football news", top_k=3)
    searcher.print_results(results)

