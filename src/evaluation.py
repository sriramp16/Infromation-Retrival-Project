

import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


class ClusterEvaluator:

    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_clustering(self, vectors: np.ndarray, labels: np.ndarray) -> Dict[str, float]:

        n_clusters = len(np.unique(labels))
        
        silhouette = silhouette_score(vectors, labels)
        
        db_index = davies_bouldin_score(vectors, labels)
        
        ch_index = calinski_harabasz_score(vectors, labels)
        
        self.metrics = {
            'silhouette_score': silhouette,
            'davies_bouldin_index': db_index,
            'calinski_harabasz_index': ch_index,
            'n_clusters': n_clusters,
            'n_documents': len(labels)
        }
        
        return self.metrics
    
    def print_evaluation(self):

        if not self.metrics:
            print("No evaluation performed yet")
            return
        
        print("\n=== Clustering Evaluation ===")
        print(f"Number of clusters: {self.metrics['n_clusters']}")
        print(f"Number of documents: {self.metrics['n_documents']}")
        print(f"\nMetrics:")
        print(f"  Silhouette Score: {self.metrics['silhouette_score']:.3f}")
        print(f"    (Higher is better, range: -1 to 1)")
        print(f"  Davies-Bouldin Index: {self.metrics['davies_bouldin_index']:.3f}")
        print(f"    (Lower is better)")
        print(f"  Calinski-Harabasz Index: {self.metrics['calinski_harabasz_index']:.1f}")
        print(f"    (Higher is better)")
    
    def plot_silhouette_analysis(self, vectors: np.ndarray, labels: np.ndarray, 
                                 figsize=(10, 6)):

        n_clusters = len(np.unique(labels))
        
        sample_silhouette_values = silhouette_samples(vectors, labels)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        y_lower = 10
        for i in range(n_clusters):
            cluster_silhouette_values = sample_silhouette_values[labels == i]
            cluster_silhouette_values.sort()
            
            size_cluster_i = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )
            
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            
            y_lower = y_upper + 10
        
        ax.set_title('Silhouette Analysis for Clustering')
        ax.set_xlabel('Silhouette Coefficient')
        ax.set_ylabel('Cluster Label')
        
        avg_score = silhouette_score(vectors, labels)
        ax.axvline(x=avg_score, color="red", linestyle="--", 
                   label=f'Average ({avg_score:.3f})')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('silhouette_analysis.png', dpi=300, bbox_inches='tight')
        print("\nSilhouette analysis saved to 'silhouette_analysis.png'")
        plt.close()


class RetrievalEvaluator:

    
    def __init__(self):
        pass
    
    def precision_at_k(self, retrieved_docs: List[Dict], 
                      relevant_doc_ids: List[int], k: int = 5) -> float:

        if k > len(retrieved_docs):
            k = len(retrieved_docs)
        
        if k == 0:
            return 0.0
        
        top_k_docs = retrieved_docs[:k]
        relevant_retrieved = sum(
            1 for doc in top_k_docs 
            if doc['document']['id'] in relevant_doc_ids
        )
        
        return relevant_retrieved / k
    
    def recall_at_k(self, retrieved_docs: List[Dict], 
                    relevant_doc_ids: List[int], k: int = 5) -> float:

        if len(relevant_doc_ids) == 0:
            return 0.0
        
        if k > len(retrieved_docs):
            k = len(retrieved_docs)
        
        top_k_docs = retrieved_docs[:k]
        relevant_retrieved = sum(
            1 for doc in top_k_docs 
            if doc['document']['id'] in relevant_doc_ids
        )
        
        return relevant_retrieved / len(relevant_doc_ids)
    
    def f1_score(self, precision: float, recall: float) -> float:

        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def mean_reciprocal_rank(self, retrieved_docs: List[Dict], 
                            relevant_doc_ids: List[int]) -> float:

        for i, doc in enumerate(retrieved_docs):
            if doc['document']['id'] in relevant_doc_ids:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def evaluate_search(self, retrieved_docs: List[Dict], 
                       relevant_doc_ids: List[int], k_values=[5, 10]) -> Dict:

        metrics = {}
        
        for k in k_values:
            precision = self.precision_at_k(retrieved_docs, relevant_doc_ids, k)
            recall = self.recall_at_k(retrieved_docs, relevant_doc_ids, k)
            f1 = self.f1_score(precision, recall)
            
            metrics[f'precision@{k}'] = precision
            metrics[f'recall@{k}'] = recall
            metrics[f'f1@{k}'] = f1
        
        metrics['mrr'] = self.mean_reciprocal_rank(retrieved_docs, relevant_doc_ids)
        
        return metrics
    
    def print_evaluation(self, metrics: Dict):

        print("\n=== Retrieval Evaluation ===")
        
        for key, value in metrics.items():
            if '@' in key:
                metric_name, k = key.split('@')
                print(f"{metric_name.capitalize()}@{k}: {value:.3f}")
            else:
                print(f"{key.upper()}: {value:.3f}")


if __name__ == "__main__":
    from preprocessing import DocumentPreprocessor
    from lsi_model import LSIModel
    from clustering import HierarchicalClustering
    
    docs = [
        {'id': 1, 'title': 'Football', 'content': 'Football match'},
        {'id': 2, 'title': 'Soccer', 'content': 'Soccer game'},
        {'id': 3, 'title': 'Election', 'content': 'Election results'}
    ]
    
    preprocessor = DocumentPreprocessor()
    processed = preprocessor.preprocess_documents(docs)
    
    lsi = LSIModel(n_topics=2)
    vectors = lsi.fit_transform(processed)
    
    clustering = HierarchicalClustering(n_clusters=2)
    labels = clustering.fit_predict(vectors)
    
    evaluator = ClusterEvaluator()
    metrics = evaluator.evaluate_clustering(vectors, labels)
    evaluator.print_evaluation()

