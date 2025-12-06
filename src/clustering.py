
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import List, Dict, Optional
import matplotlib.pyplot as plt


class HierarchicalClustering:

    
    def __init__(self, n_clusters=5, linkage_method='ward', distance_threshold=None):

        self.n_clusters = n_clusters if distance_threshold is None else None
        self.linkage_method = linkage_method
        self.distance_threshold = distance_threshold
        
        self.model = None
        self.labels_ = None
        self.linkage_matrix = None
    
    def fit(self, vectors: np.ndarray) -> 'HierarchicalClustering':

        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage_method,
            distance_threshold=self.distance_threshold,
            compute_full_tree=True
        )
        
        self.labels_ = self.model.fit_predict(vectors)
        
        self.linkage_matrix = self._compute_linkage_matrix(vectors)
        
        print(f"\nHierarchical Clustering Results:")
        print(f"  - Number of clusters: {len(np.unique(self.labels_))}")
        print(f"  - Cluster sizes: {np.bincount(self.labels_)}")
        
        return self
    
    def fit_predict(self, vectors: np.ndarray) -> np.ndarray:

        self.fit(vectors)
        return self.labels_
    
    def predict_single(self, vector: np.ndarray, cluster_centers: np.ndarray) -> int:

        distances = np.linalg.norm(cluster_centers - vector, axis=1)
        return np.argmin(distances)
    
    def _compute_linkage_matrix(self, vectors: np.ndarray) -> np.ndarray:

        return linkage(vectors, method=self.linkage_method)
    
    def get_cluster_centers(self, vectors: np.ndarray) -> np.ndarray:

        if self.labels_ is None:
            raise ValueError("Model not fitted yet")
        
        n_clusters = len(np.unique(self.labels_))
        centers = np.zeros((n_clusters, vectors.shape[1]))
        
        for cluster_id in range(n_clusters):
            cluster_mask = self.labels_ == cluster_id
            centers[cluster_id] = vectors[cluster_mask].mean(axis=0)
        
        return centers
    
    def get_cluster_documents(self, processed_docs: List[Dict]) -> Dict[int, List[Dict]]:

        if self.labels_ is None:
            raise ValueError("Model not fitted yet")
        
        clusters = {}
        for cluster_id in np.unique(self.labels_):
            cluster_mask = self.labels_ == cluster_id
            cluster_docs = [doc for i, doc in enumerate(processed_docs) if cluster_mask[i]]
            clusters[cluster_id] = cluster_docs
        
        return clusters
    
    def plot_dendrogram(self, labels: Optional[List[str]] = None, figsize=(12, 6)):

        if self.linkage_matrix is None:
            raise ValueError("Model not fitted yet")
        
        plt.figure(figsize=figsize)
        
        dendrogram(
            self.linkage_matrix,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=8
        )
        
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Document Index' if labels is None else 'Document')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
        print("\nDendrogram saved to 'dendrogram.png'")
        plt.close()
    
    def print_cluster_summary(self, processed_docs: List[Dict], n_terms=5):

        if self.labels_ is None:
            raise ValueError("Model not fitted yet")
        
        clusters = self.get_cluster_documents(processed_docs)
        
        print("\n=== Cluster Summary ===")
        for cluster_id, docs in sorted(clusters.items()):
            print(f"\nCluster {cluster_id} ({len(docs)} documents):")
            
            all_tokens = []
            for doc in docs:
                all_tokens.extend(doc['tokens'])
            
            from collections import Counter
            term_freq = Counter(all_tokens)
            top_terms = term_freq.most_common(n_terms)
            
            print(f"  Top terms: {', '.join([term for term, _ in top_terms])}")
            
            sample_titles = [doc['title'] for doc in docs[:3]]
            print(f"  Sample docs: {', '.join(sample_titles)}")


if __name__ == "__main__":
    from preprocessing import DocumentPreprocessor
    from lsi_model import LSIModel
    
    sample_docs = [
        {'id': 1, 'title': 'Football Match', 'content': 'Football victory'},
        {'id': 2, 'title': 'Soccer Game', 'content': 'Soccer highlights'},
        {'id': 3, 'title': 'Election', 'content': 'Election results'},
        {'id': 4, 'title': 'Politics', 'content': 'Political debate'},
        {'id': 5, 'title': 'AI Tech', 'content': 'New AI technology'}
    ]
    
    preprocessor = DocumentPreprocessor()
    processed = preprocessor.preprocess_documents(sample_docs)
    
    lsi = LSIModel(n_topics=3)
    lsi_vectors = lsi.fit_transform(processed)
    
    clustering = HierarchicalClustering(n_clusters=3)
    labels = clustering.fit_predict(lsi_vectors)
    
    clustering.print_cluster_summary(processed)

