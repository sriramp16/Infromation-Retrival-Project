

import numpy as np
from typing import List, Dict, Tuple
from src.preprocessing import DocumentPreprocessor
from src.lsi_model import LSIModel
from src.clustering import HierarchicalClustering


class DynamicClusterUpdater:

    
    def __init__(self, preprocessor: DocumentPreprocessor, 
                 lsi_model: LSIModel, 
                 clustering: HierarchicalClustering):

        self.preprocessor = preprocessor
        self.lsi_model = lsi_model
        self.clustering = clustering
        
        self.all_documents = []
        self.all_vectors = None
        self.all_labels = None
    
    def initialize(self, processed_docs: List[Dict], vectors: np.ndarray, labels: np.ndarray):

        self.all_documents = processed_docs.copy()
        self.all_vectors = vectors.copy()
        self.all_labels = labels.copy()
        
        print(f"Dynamic updater initialized with {len(processed_docs)} documents")
    
    def add_document(self, new_doc: Dict[str, str], 
                     refit_threshold=10) -> Tuple[int, Dict]:

        processed = self.preprocessor.preprocess_documents([new_doc])[0]
        
        new_vector = self.lsi_model.transform([processed])[0]
        
        cluster_centers = self.clustering.get_cluster_centers(self.all_vectors)
        
        cluster_label = self.clustering.predict_single(new_vector, cluster_centers)
        
        self.all_documents.append(processed)
        self.all_vectors = np.vstack([self.all_vectors, new_vector])
        self.all_labels = np.append(self.all_labels, cluster_label)
        
        print(f"\nAdded document '{processed['title']}' to Cluster {cluster_label}")
        
        num_new_docs = len(self.all_documents) - len(self.clustering.labels_)
        if refit_threshold > 0 and num_new_docs >= refit_threshold:
            print(f"  Refitting clusters ({num_new_docs} new documents added)")
            self._refit_clusters()
        
        return cluster_label, processed
    
    def add_documents_batch(self, new_docs: List[Dict[str, str]]) -> List[int]:

        labels = []
        
        for doc in new_docs:
            label, _ = self.add_document(doc, refit_threshold=0)
            labels.append(label)
        
        if len(new_docs) >= 5:
            print(f"\nRefitting clusters after adding {len(new_docs)} documents")
            self._refit_clusters()
        
        return labels
    
    def _refit_clusters(self):

        self.all_labels = self.clustering.fit_predict(self.all_vectors)
        
        print(f"  Refitted with {len(self.all_documents)} total documents")
        print(f"  New cluster sizes: {np.bincount(self.all_labels)}")
    
    def get_cluster_summary(self, cluster_id: int) -> Dict:

        cluster_mask = self.all_labels == cluster_id
        cluster_docs = [doc for i, doc in enumerate(self.all_documents) if cluster_mask[i]]
        
        all_tokens = []
        for doc in cluster_docs:
            all_tokens.extend(doc['tokens'])
        
        from collections import Counter
        term_freq = Counter(all_tokens)
        top_terms = term_freq.most_common(10)
        
        return {
            'cluster_id': cluster_id,
            'num_documents': len(cluster_docs),
            'top_terms': [term for term, _ in top_terms],
            'documents': cluster_docs
        }
    
    def get_all_data(self) -> Tuple[List[Dict], np.ndarray, np.ndarray]:

        return self.all_documents, self.all_vectors, self.all_labels


if __name__ == "__main__":
    from preprocessing import DocumentPreprocessor
    from lsi_model import LSIModel
    from clustering import HierarchicalClustering
    
    initial_docs = [
        {'id': 1, 'title': 'Football', 'content': 'Football match victory'},
        {'id': 2, 'title': 'Soccer', 'content': 'Soccer game highlights'},
        {'id': 3, 'title': 'Election', 'content': 'Election results'}
    ]
    
    preprocessor = DocumentPreprocessor()
    processed = preprocessor.preprocess_documents(initial_docs)
    
    lsi = LSIModel(n_topics=2)
    vectors = lsi.fit_transform(processed)
    
    clustering = HierarchicalClustering(n_clusters=2)
    labels = clustering.fit_predict(vectors)
    
    updater = DynamicClusterUpdater(preprocessor, lsi, clustering)
    updater.initialize(processed, vectors, labels)
    
    new_doc = {'id': 4, 'title': 'Championship', 'content': 'Football championship news'}
    cluster, _ = updater.add_document(new_doc)
    print(f"New document assigned to cluster {cluster}")

