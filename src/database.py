import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json
import pickle
import numpy as np


class IRDatabase:
    def __init__(self, db_path='ir_system.db', use_postgres=False, postgres_config=None):
        self.db_path = db_path
        self.use_postgres = use_postgres
        self.postgres_config = postgres_config
        self.conn = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        if not self.use_postgres:
            self.conn = sqlite3.connect(self.db_path)
        else:
            try:
                import psycopg2
                self.conn = psycopg2.connect(**self.postgres_config)
            except ImportError:
                print("PostgreSQL not available, using SQLite")
                self.use_postgres = False
                self.conn = sqlite3.connect(self.db_path)
    
    def _create_tables(self):
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id INTEGER PRIMARY KEY,
                size INTEGER,
                centroid BLOB,
                top_terms TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_clusters (
                doc_id INTEGER,
                cluster_id INTEGER,
                probability REAL,
                PRIMARY KEY (doc_id, cluster_id),
                FOREIGN KEY (doc_id) REFERENCES documents(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                result_count INTEGER,
                execution_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lsi_vectors (
                doc_id INTEGER PRIMARY KEY,
                vector BLOB,
                FOREIGN KEY (doc_id) REFERENCES documents(id)
            )
        ''')
        
        self.conn.commit()
    
    def insert_document(self, doc: Dict) -> int:
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO documents (title, content, category, source)
                VALUES (?, ?, ?, ?)
            ''', (doc.get('title'), doc.get('content'), doc.get('category'), doc.get('source')))
            
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"Error inserting document: {e}")
            return -1
    
    def get_document(self, doc_id: int) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
        row = cursor.fetchone()
        
        if row:
            return {
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'category': row[3],
                'source': row[4],
                'created_at': row[5],
                'updated_at': row[6]
            }
        return None
    
    def get_all_documents(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM documents')
        rows = cursor.fetchall()
        
        docs = []
        for row in rows:
            docs.append({
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'category': row[3],
                'source': row[4],
                'created_at': row[5],
                'updated_at': row[6]
            })
        return docs
    
    def update_document(self, doc_id: int, doc: Dict) -> bool:
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                UPDATE documents 
                SET title = ?, content = ?, category = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (doc.get('title'), doc.get('content'), doc.get('category'), doc_id))
            
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating document: {e}")
            return False
    
    def delete_document(self, doc_id: int) -> bool:
        cursor = self.conn.cursor()
        try:
            cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
            cursor.execute('DELETE FROM document_clusters WHERE doc_id = ?', (doc_id,))
            cursor.execute('DELETE FROM lsi_vectors WHERE doc_id = ?', (doc_id,))
            
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def insert_cluster(self, cluster_id: int, size: int, centroid=None, top_terms=None) -> bool:
        cursor = self.conn.cursor()
        try:
            centroid_blob = None
            if centroid is not None:
                centroid_blob = pickle.dumps(centroid)
            
            terms_str = json.dumps(top_terms) if top_terms else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO clusters (cluster_id, size, centroid, top_terms)
                VALUES (?, ?, ?, ?)
            ''', (cluster_id, size, centroid_blob, terms_str))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error inserting cluster: {e}")
            return False
    
    def get_cluster(self, cluster_id: int) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM clusters WHERE cluster_id = ?', (cluster_id,))
        row = cursor.fetchone()
        
        if row:
            centroid = None
            if row[2]:
                centroid = pickle.loads(row[2])
            
            top_terms = []
            if row[3]:
                top_terms = json.loads(row[3])
            
            return {
                'cluster_id': row[0],
                'size': row[1],
                'centroid': centroid,
                'top_terms': top_terms,
                'created_at': row[4]
            }
        return None
    
    def insert_document_cluster(self, doc_id: int, cluster_id: int, probability: float = 1.0) -> bool:
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO document_clusters (doc_id, cluster_id, probability)
                VALUES (?, ?, ?)
            ''', (doc_id, cluster_id, probability))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error inserting document-cluster: {e}")
            return False
    
    def get_cluster_documents(self, cluster_id: int) -> List[int]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT doc_id FROM document_clusters WHERE cluster_id = ?', (cluster_id,))
        rows = cursor.fetchall()
        return [row[0] for row in rows]
    
    def log_search(self, query: str, result_count: int, execution_time: float) -> bool:
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO search_logs (query, result_count, execution_time)
                VALUES (?, ?, ?)
            ''', (query, result_count, execution_time))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error logging search: {e}")
            return False
    
    def insert_lsi_vector(self, doc_id: int, vector) -> bool:
        cursor = self.conn.cursor()
        try:
            vector_blob = pickle.dumps(vector)
            cursor.execute('''
                INSERT OR REPLACE INTO lsi_vectors (doc_id, vector)
                VALUES (?, ?)
            ''', (doc_id, vector_blob))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error inserting LSI vector: {e}")
            return False
    
    def close(self):
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        self.close()

