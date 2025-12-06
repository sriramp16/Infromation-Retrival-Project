from flask import Flask, render_template, request, jsonify
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing import DocumentPreprocessor
from src.lsi_model import LSIModel
from src.clustering import HierarchicalClustering
from src.search import DocumentSearcher
from src.evaluation import ClusterEvaluator, RetrievalEvaluator
from src.database import IRDatabase
from src.dynamic_update import DynamicClusterUpdater
from data.sample_documents import get_sample_documents
from collections import Counter
import io


app = Flask(__name__)
app.config['SECRET_KEY'] = 'ir-system-secret-key'

global_state = {
    'preprocessor': None,
    'lsi_model': None,
    'clustering': None,
    'searcher': None,
    'updater': None,
    'db': None,
    'documents': [],
    'processed_docs': [],
    'vectors': None,
    'labels': None,
    'initialized': False
}


def initialize_system():
    print("Initializing IR System...")
    
    global_state['preprocessor'] = DocumentPreprocessor(use_stemming=True)
    global_state['lsi_model'] = LSIModel(n_topics=5)
    global_state['clustering'] = HierarchicalClustering(n_clusters=3)
    global_state['db'] = IRDatabase()
    
    global_state['initialized'] = True
    print("✓ System initialized")


def load_documents_and_build():
    if not global_state['initialized']:
        initialize_system()
    
    if len(global_state['documents']) > 0:
        print("Documents already loaded")
        return
    
    print("Loading sample documents...")
    docs = get_sample_documents()
    global_state['documents'] = docs
    
    print("Preprocessing documents...")
    global_state['processed_docs'] = global_state['preprocessor'].preprocess_documents(docs)
    
    print("Building LSI model...")
    global_state['vectors'] = global_state['lsi_model'].fit_transform(global_state['processed_docs'])
    
    print("Creating clusters...")
    global_state['labels'] = global_state['clustering'].fit_predict(global_state['vectors'])
    
    print("Indexing for search...")
    global_state['searcher'] = DocumentSearcher(
        global_state['preprocessor'],
        global_state['lsi_model']
    )
    global_state['searcher'].index_documents(
        global_state['processed_docs'],
        global_state['vectors'],
        global_state['labels']
    )
    
    global_state['updater'] = DynamicClusterUpdater(
        global_state['preprocessor'],
        global_state['lsi_model'],
        global_state['clustering']
    )
    global_state['updater'].initialize(
        global_state['processed_docs'],
        global_state['vectors'],
        global_state['labels']
    )
    
    print("✓ System ready")


def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        import pdfplumber
        
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num}: {e}")
                    continue
        
        if text_parts:
            return '\n\n'.join(text_parts)
        
    except ImportError:
        print("pdfplumber not available, trying PyPDF2...")
    
    try:
        import PyPDF2
        
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_parts = []
        for page_num in range(len(pdf_reader.pages)):
            try:
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            except Exception as e:
                print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                continue
        
        if text_parts:
            return '\n\n'.join(text_parts)
            
    except ImportError:
        raise ImportError("No PDF extraction library available. Please install pdfplumber or PyPDF2.")
    except Exception as e:
        print(f"Error with PyPDF2: {e}")
    
    raise ValueError("Could not extract text from PDF. The PDF may be encrypted, image-based, or corrupted.")


@app.route('/')
def index():
    if not global_state['initialized']:
        initialize_system()
    
    if len(global_state['documents']) == 0:
        load_documents_and_build()
    
    stats = {
        'total_docs': len(global_state['documents']),
        'n_clusters': len(set(global_state['labels'])) if global_state['labels'] is not None else 0,
        'n_topics': global_state['lsi_model'].n_topics if global_state['lsi_model'] else 0
    }
    
    return render_template('index.html', stats=stats)


@app.route('/api/search')
def api_search():
    query = request.args.get('q', '')
    top_k = int(request.args.get('k', 10))
    category_filter = request.args.get('category', None)
    
    if not query or not global_state['searcher']:
        return jsonify({'results': []})
    
    results = global_state['searcher'].search(query, top_k=top_k, category_filter=category_filter)
    
    formatted = []
    for result in results:
        formatted.append({
            'rank': result['rank'],
            'title': result['document']['title'],
            'content': result['document'].get('original_content', '')[:200] + '...',
            'score': round(result['score'], 3),
            'cluster': result.get('cluster'),
            'category': result['document'].get('category', ''),
            'doc_id': result['document']['id']
        })
    
    return jsonify({'query': query, 'results': formatted})


@app.route('/clusters')
def clusters():
    if not global_state['initialized']:
        initialize_system()
    
    if len(global_state['documents']) == 0:
        load_documents_and_build()
    
    unique_labels = set(global_state['labels'])
    cluster_info = []
    
    for cluster_id in unique_labels:
        cluster_docs = []
        for i, label in enumerate(global_state['labels']):
            if label == cluster_id:
                cluster_docs.append({
                    'id': global_state['documents'][i]['id'],
                    'title': global_state['documents'][i]['title'],
                    'content': global_state['documents'][i]['content'][:150] + '...'
                })
        
        all_tokens = []
        for i, label in enumerate(global_state['labels']):
            if label == cluster_id:
                all_tokens.extend(global_state['processed_docs'][i]['tokens'])
        
        top_terms = [term for term, _ in Counter(all_tokens).most_common(10)]
        
        cluster_info.append({
            'id': int(cluster_id),
            'size': len(cluster_docs),
            'top_terms': top_terms,
            'documents': cluster_docs[:10]
        })
    
    cluster_info.sort(key=lambda x: x['id'])
    
    return render_template('clusters.html', clusters=cluster_info)


@app.route('/cluster/<int:cluster_id>')
def cluster_detail(cluster_id):
    if not global_state['initialized'] or global_state['labels'] is None:
        return "System not initialized", 404
    
    cluster_docs = []
    for i, label in enumerate(global_state['labels']):
        if label == cluster_id:
            cluster_docs.append(global_state['documents'][i])
    
    if len(cluster_docs) == 0:
        return "Cluster not found", 404
    
    all_tokens = []
    for i, label in enumerate(global_state['labels']):
        if label == cluster_id:
            all_tokens.extend(global_state['processed_docs'][i]['tokens'])
    
    top_terms = [term for term, _ in Counter(all_tokens).most_common(20)]
    
    return render_template('cluster_detail.html',
                         cluster_id=cluster_id,
                         documents=cluster_docs,
                         top_terms=top_terms)


@app.route('/analytics')
def analytics():
    if not global_state['initialized'] or global_state['labels'] is None:
        initialize_system()
    
    if len(global_state['documents']) == 0:
        load_documents_and_build()
    
    evaluator = ClusterEvaluator()
    if global_state['vectors'] is not None and global_state['labels'] is not None:
        metrics = evaluator.evaluate_clustering(global_state['vectors'], global_state['labels'])
    else:
        metrics = {'silhouette_score': 0, 'davies_bouldin_index': 0, 'calinski_harabasz_index': 0}
    
    cluster_sizes = dict(Counter(global_state['labels']))
    
    cluster_breakdown = []
    for cluster_id, size in sorted(cluster_sizes.items()):
        cluster_breakdown.append({
            'cluster_id': int(cluster_id),
            'size': size
        })
    
    return render_template('analytics.html',
                         metrics=metrics,
                         cluster_breakdown=cluster_breakdown,
                         total_docs=len(global_state['documents']))


@app.route('/api/evaluate')
def api_evaluate():
    if global_state['vectors'] is None or global_state['labels'] is None:
        return jsonify({'error': 'System not built'}), 400
    
    evaluator = ClusterEvaluator()
    metrics = evaluator.evaluate_clustering(global_state['vectors'], global_state['labels'])
    
    return jsonify(metrics)


@app.route('/upload_document')
def upload_document():
    return render_template('upload_document.html')


@app.route('/api/add_document', methods=['POST'])
def api_add_document():
    data = request.json
    
    if not data or 'title' not in data or 'content' not in data:
        return jsonify({'error': 'Title and content are required'}), 400
    
    if not global_state['initialized']:
        initialize_system()
    
    new_doc = {
        'id': len(global_state['documents']) + 1,
        'title': data['title'],
        'content': data['content'],
        'source': data.get('source', 'Manual')
    }
    
    global_state['documents'].append(new_doc)
    
    try:
        global_state['processed_docs'] = global_state['preprocessor'].preprocess_documents(global_state['documents'])
        
        global_state['vectors'] = global_state['lsi_model'].fit_transform(global_state['processed_docs'])
        
        global_state['labels'] = global_state['clustering'].fit_predict(global_state['vectors'])
        
        global_state['searcher'] = DocumentSearcher(
            global_state['preprocessor'],
            global_state['lsi_model']
        )
        global_state['searcher'].index_documents(
            global_state['processed_docs'],
            global_state['vectors'],
            global_state['labels']
        )
        
        return jsonify({
            'success': True,
            'message': f'Document added successfully. Total documents: {len(global_state["documents"])}',
            'total_docs': len(global_state['documents']),
            'n_clusters': len(set(global_state['labels'])),
            'n_topics': global_state['lsi_model'].n_topics
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to rebuild system: {str(e)}'}), 500


@app.route('/api/add_multiple_documents', methods=['POST'])
def api_add_multiple_documents():
    data = request.json
    
    if not data or 'documents' not in data:
        return jsonify({'error': 'Documents array is required'}), 400
    
    documents = data['documents']
    if not isinstance(documents, list) or len(documents) == 0:
        return jsonify({'error': 'Documents must be a non-empty array'}), 400
    
    if not global_state['initialized']:
        initialize_system()
    
    for i, doc in enumerate(documents):
        if 'title' not in doc or 'content' not in doc:
            return jsonify({'error': f'Document {i+1} missing title or content'}), 400
        
        new_doc = {
            'id': len(global_state['documents']) + 1,
            'title': doc['title'],
            'content': doc['content'],
            'source': doc.get('source', 'Manual')
        }
        global_state['documents'].append(new_doc)
    
    try:
        global_state['processed_docs'] = global_state['preprocessor'].preprocess_documents(global_state['documents'])
        
        global_state['vectors'] = global_state['lsi_model'].fit_transform(global_state['processed_docs'])
        
        global_state['labels'] = global_state['clustering'].fit_predict(global_state['vectors'])
        
        global_state['searcher'] = DocumentSearcher(
            global_state['preprocessor'],
            global_state['lsi_model']
        )
        global_state['searcher'].index_documents(
            global_state['processed_docs'],
            global_state['vectors'],
            global_state['labels']
        )
        
        return jsonify({
            'success': True,
            'message': f'Added {len(documents)} documents successfully. Total documents: {len(global_state["documents"])}',
            'total_docs': len(global_state['documents']),
            'n_clusters': len(set(global_state['labels'])),
            'n_topics': global_state['lsi_model'].n_topics
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to rebuild system: {str(e)}'}), 500


@app.route('/api/rebuild_system', methods=['POST'])
def api_rebuild_system():
    if not global_state['initialized']:
        return jsonify({'error': 'System not initialized'}), 400
    
    if len(global_state['documents']) == 0:
        return jsonify({'error': 'No documents to rebuild'}), 400
    
    try:
        global_state['processed_docs'] = global_state['preprocessor'].preprocess_documents(global_state['documents'])
        global_state['vectors'] = global_state['lsi_model'].fit_transform(global_state['processed_docs'])
        global_state['labels'] = global_state['clustering'].fit_predict(global_state['vectors'])
        
        global_state['searcher'] = DocumentSearcher(
            global_state['preprocessor'],
            global_state['lsi_model']
        )
        global_state['searcher'].index_documents(
            global_state['processed_docs'],
            global_state['vectors'],
            global_state['labels']
        )
        
        return jsonify({
            'success': True,
            'message': 'System rebuilt successfully',
            'total_docs': len(global_state['documents']),
            'n_clusters': len(set(global_state['labels'])),
            'n_topics': global_state['lsi_model'].n_topics
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to rebuild system: {str(e)}'}), 500


@app.route('/api/upload_document', methods=['POST'])
def api_upload_document():
    from werkzeug.utils import secure_filename
    
    if not global_state['initialized']:
        initialize_system()
    
    if len(global_state['documents']) == 0:
        load_documents_and_build()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        filename = secure_filename(file.filename)
        file_content = file.read()
        
        file_type = 'text'
        pdf_pages = 0
        
        if filename.lower().endswith(('.txt', '.text', '.md')):
            text_content = file_content.decode('utf-8', errors='ignore')
        elif filename.lower().endswith('.pdf'):
            file_type = 'pdf'
            try:
                text_content = extract_text_from_pdf(file_content)
                if not text_content or len(text_content.strip()) < 50:
                    return jsonify({'error': 'Could not extract meaningful text from PDF. The PDF may be image-based or corrupted.'}), 400
                
                try:
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                        pdf_pages = len(pdf.pages)
                except:
                    try:
                        import PyPDF2
                        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                        pdf_pages = len(pdf_reader.pages)
                    except:
                        pass
                        
            except Exception as e:
                return jsonify({'error': f'Failed to extract text from PDF: {str(e)}'}), 400
        else:
            return jsonify({'error': 'Unsupported file type. Supported formats: .txt, .text, .md, .pdf'}), 400
        
        title = request.form.get('title', '').strip()
        if not title:
            title = os.path.splitext(filename)[0]
        
        new_doc = {
            'id': len(global_state['documents']) + 1,
            'title': title,
            'content': text_content,
            'source': 'Uploaded File'
        }
        
        original_length = len(text_content)
        
        processed = global_state['preprocessor'].preprocess_documents([new_doc])[0]
        num_tokens = len(processed['tokens'])
        unique_terms = len(set(processed['tokens']))
        stopwords_removed = original_length - len(' '.join(processed['tokens']))
        
        token_counter = Counter(processed['tokens'])
        top_tokens = [token for token, _ in token_counter.most_common(20)]
        
        global_state['documents'].append(new_doc)
        
        global_state['processed_docs'] = global_state['preprocessor'].preprocess_documents(
            global_state['documents']
        )
        global_state['vectors'] = global_state['lsi_model'].fit_transform(
            global_state['processed_docs']
        )
        global_state['labels'] = global_state['clustering'].fit_predict(
            global_state['vectors']
        )
        
        cluster_id = global_state['labels'][-1]
        
        global_state['searcher'].index_documents(
            global_state['processed_docs'],
            global_state['vectors'],
            global_state['labels']
        )
        
        evaluator = ClusterEvaluator()
        metrics = evaluator.evaluate_clustering(global_state['vectors'], global_state['labels'])
        
        response = {
            'document': {
                'title': new_doc['title'],
                'filename': filename,
                'file_type': file_type,
                'pdf_pages': pdf_pages if file_type == 'pdf' else None,
                'cluster_id': int(cluster_id)
            },
            'preprocessing': {
                'original_length': original_length,
                'num_tokens': num_tokens,
                'unique_terms': unique_terms,
                'stopwords_removed': max(0, stopwords_removed),
                'use_stemming': global_state['preprocessor'].use_stemming,
                'top_tokens': top_tokens
            },
            'lsi': {
                'n_topics': global_state['lsi_model'].n_topics,
                'vector_shape': list(global_state['vectors'][-1].shape),
                'explained_variance': getattr(global_state['lsi_model'], 'explained_variance_ratio_', 0)
            },
            'clustering': {
                'cluster_id': int(cluster_id),
                'total_clusters': len(set(global_state['labels']))
            },
            'evaluation': {
                'silhouette_score': float(metrics['silhouette_score']),
                'davies_bouldin_index': float(metrics['davies_bouldin_index']),
                'calinski_harabasz_index': float(metrics['calinski_harabasz_index']),
                'total_documents': int(metrics['n_documents']),
                'n_clusters': int(metrics['n_clusters'])
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to process document: {str(e)}'}), 500


if __name__ == '__main__':
    initialize_system()
    
    print("\n" + "=" * 60)
    print("IR System Web Application")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Visit: http://localhost:5000")
    print("\nRoutes:")
    print("  /                    - Homepage with search")
    print("  /clusters            - Cluster visualization")
    print("  /analytics           - Statistics dashboard")
    print("  /upload_document     - Upload document for processing")
    print("  /api/search          - Search API")
    print("  /api/add_document     - Add single document")
    print("  /api/upload_document  - Upload and process document file")
    print("  /api/add_multiple_documents - Add multiple documents")
    print("  /api/rebuild_system  - Rebuild LSI and clusters")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
