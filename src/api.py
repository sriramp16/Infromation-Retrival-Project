

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import logging

from src.preprocessing import DocumentPreprocessor
from src.lsi_model import LSIModel
from src.clustering import HierarchicalClustering
from src.dynamic_update import DynamicClusterUpdater
from src.search import DocumentSearcher
from src.evaluation import ClusterEvaluator, RetrievalEvaluator
from src.database import IRDatabase


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dynamic Web IR System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

system_state = {
    'preprocessor': None,
    'lsi_model': None,
    'clustering': None,
    'updater': None,
    'searcher': None,
    'documents': [],
    'vectors': None,
    'labels': None,
    'db': IRDatabase(),
    'initialized': False
}


class DocumentInput(BaseModel):
    id: int
    title: str
    content: str
    url: Optional[str] = None
    author: Optional[str] = None
    published: Optional[str] = None
    source: Optional[str] = None


class QueryInput(BaseModel):
    query: str
    top_k: int = 10
    cluster_filter: Optional[int] = None


class SystemConfig(BaseModel):
    n_topics: int = 10
    n_clusters: int = 5
    use_stemming: bool = True
    linkage_method: str = 'ward'


class ClusterRequest(BaseModel):
    n_clusters: Optional[int] = None
    linkage_method: str = 'ward'



@app.get("/")
async def root():
Health check endpoint"""

    return {
        "status": "healthy",
        "initialized": system_state['initialized']
    }


@app.post("/initialize")
async def initialize_system(config: SystemConfig):

    try:
        system_state['preprocessor'] = DocumentPreprocessor(
            use_stemming=config.use_stemming
        )
        system_state['lsi_model'] = LSIModel(n_topics=config.n_topics)
        system_state['clustering'] = HierarchicalClustering(
            n_clusters=config.n_clusters,
            linkage_method=config.linkage_method
        )
        system_state['initialized'] = True
        
        logger.info("System initialized")
        return {"message": "System initialized successfully", "config": config.dict()}
    
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents")
async def add_documents(documents: List[DocumentInput]):

    if not system_state['initialized']:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    try:
        docs = [doc.dict() for doc in documents]
        
        for doc in docs:
            system_state['db'].insert_document(doc)
        
        system_state['documents'].extend(docs)
        
        logger.info(f"Added {len(docs)} documents")
        return {
            "message": f"Added {len(docs)} documents",
            "total_documents": len(system_state['documents'])
        }
    
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/build")
async def build_clusters():

    if not system_state['initialized']:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    if not system_state['documents']:
        raise HTTPException(status_code=400, detail="No documents to cluster")
    
    try:
        processed_docs = system_state['preprocessor'].preprocess_documents(
            system_state['documents']
        )
        
        system_state['vectors'] = system_state['lsi_model'].fit_transform(processed_docs)
        
        system_state['labels'] = system_state['clustering'].fit_predict(
            system_state['vectors']
        )
        
        system_state['searcher'] = DocumentSearcher(
            system_state['preprocessor'],
            system_state['lsi_model']
        )
        system_state['searcher'].index_documents(
            processed_docs, system_state['vectors'], system_state['labels']
        )
        
        system_state['updater'] = DynamicClusterUpdater(
            system_state['preprocessor'],
            system_state['lsi_model'],
            system_state['clustering']
        )
        system_state['updater'].initialize(
            processed_docs,
            system_state['vectors'],
            system_state['labels']
        )
        
        for i, label in enumerate(system_state['labels']):
            db_doc_id = i + 1  # Assuming sequential IDs
            system_state['db'].assign_to_cluster(db_doc_id, int(label))
        
        n_clusters = len(np.unique(system_state['labels']))
        cluster_sizes = np.bincount(system_state['labels']).tolist()
        
        logger.info(f"Built {n_clusters} clusters")
        return {
            "message": "Clusters built successfully",
            "n_documents": len(system_state['documents']),
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes
        }
    
    except Exception as e:
        logger.error(f"Error building clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search(query: QueryInput):

    if not system_state['initialized'] or not system_state['searcher']:
        raise HTTPException(status_code=400, detail="System not built")
    
    try:
        results = system_state['searcher'].search(
            query.query,
            top_k=query.top_k,
            cluster_filter=query.cluster_filter
        )
        
        system_state['db'].log_query(query.query, len(results))
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                'rank': result['rank'],
                'title': result['document']['title'],
                'content': result['content'][:500],  # Truncate for response
                'score': result['score'],
                'cluster': result['cluster'],
                'doc_id': result['document'].get('id')
            })
        
        return {
            "query": query.query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }
    
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clusters")
async def get_clusters():

    if not system_state['labels'] is None:
        raise HTTPException(status_code=400, detail="Clusters not built yet")
    
    try:
        clusters_info = []
        unique_labels = np.unique(system_state['labels'])
        
        for cluster_id in unique_labels:
            mask = system_state['labels'] == cluster_id
            cluster_docs = [system_state['documents'][i] for i in range(len(mask)) if mask[i]]
            
            metadata = system_state['db'].get_cluster_metadata(int(cluster_id))
            
            clusters_info.append({
                'cluster_id': int(cluster_id),
                'num_documents': len(cluster_docs),
                'top_terms': metadata.get('top_terms', []) if metadata else [],
                'documents': [
                    {
                        'id': doc['id'],
                        'title': doc['title']
                    }
                    for doc in cluster_docs[:5]  # Limit for API response
                ]
            })
        
        return {
            "n_clusters": len(unique_labels),
            "clusters": clusters_info
        }
    
    except Exception as e:
        logger.error(f"Error getting clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cluster/{cluster_id}")
async def get_cluster_details(cluster_id: int):

    if system_state['labels'] is None:
        raise HTTPException(status_code=400, detail="Clusters not built yet")
    
    try:
        mask = system_state['labels'] == cluster_id
        cluster_docs = [
            system_state['documents'][i]
            for i in range(len(mask))
            if mask[i]
        ]
        
        metadata = system_state['db'].get_cluster_metadata(cluster_id)
        
        return {
            'cluster_id': cluster_id,
            'num_documents': len(cluster_docs),
            'documents': cluster_docs,
            'metadata': metadata
        }
    
    except Exception as e:
        logger.error(f"Error getting cluster: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/topics")
async def get_lsi_topics():

    if not system_state['initialized'] or system_state['lsi_model'].lsi_matrix is None:
        raise HTTPException(status_code=400, detail="LSI model not built")
    
    try:
        topics = system_state['lsi_model'].get_top_terms_per_topic(n_terms=10)
        
        formatted_topics = []
        for topic_id, terms in topics.items():
            formatted_topics.append({
                'topic_id': topic_id,
                'top_terms': [{'term': term, 'weight': weight} for term, weight in terms]
            })
        
        return {
            'n_topics': len(topics),
            'topics': formatted_topics
        }
    
    except Exception as e:
        logger.error(f"Error getting topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_document")
async def add_single_document(document: DocumentInput):

    if not system_state['initialized'] or system_state['updater'] is None:
        raise HTTPException(status_code=400, detail="System not initialized or built")
    
    try:
        doc_dict = document.dict()
        
        cluster_id, processed = system_state['updater'].add_document(doc_dict)
        
        system_state['documents'], system_state['vectors'], system_state['labels'] = \
            system_state['updater'].get_all_data()
        system_state['searcher'].index_documents(
            system_state['documents'],
            system_state['vectors'],
            system_state['labels']
        )
        
        db_id = system_state['db'].insert_document(doc_dict)
        system_state['db'].assign_to_cluster(db_id, cluster_id)
        
        return {
            "message": "Document added successfully",
            "cluster_id": int(cluster_id),
            "total_documents": len(system_state['documents'])
        }
    
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics")
async def get_statistics():

    stats = system_state['db'].get_statistics()
    
    stats['initialized'] = system_state['initialized']
    stats['has_vectors'] = system_state['vectors'] is not None
    stats['has_clusters'] = system_state['labels'] is not None
    
    return stats


@app.get("/evaluate/clustering")
async def evaluate_clustering():

    if system_state['vectors'] is None or system_state['labels'] is None:
        raise HTTPException(status_code=400, detail="Clusters not built yet")
    
    try:
        evaluator = ClusterEvaluator()
        metrics = evaluator.evaluate_clustering(
            system_state['vectors'],
            system_state['labels']
        )
        
        return {
            'metrics': metrics
        }
    
    except Exception as e:
        logger.error(f"Error evaluating clustering: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

