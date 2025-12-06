# Hierarchical Clustering with LSI for Dynamic Web IR

A comprehensive Information Retrieval system that organizes documents into hierarchical topic clusters using Latent Semantic Indexing (LSI) and Hierarchical Agglomerative Clustering (HAC).

## Features

### Core IR System
- ✅ **Document Preprocessing**: Tokenization, stemming, stop-word removal (Unit 1)
- ✅ **Latent Semantic Indexing**: Discovers hidden semantic relationships (Unit 4)
- ✅ **Hierarchical Clustering**: Creates tree-like topic hierarchies (Unit 4)
- ✅ **Dynamic Updates**: Adds new documents to existing clusters (Unit 5)
- ✅ **Search & Evaluation**: Query processing with precision and silhouette metrics (Unit 2, 4)

### Web Applications
- ✅ **Flask Web App**: Simple HTML interface for search, clusters, and analytics
- ✅ **FastAPI Backend**: Production-ready REST API with Swagger documentation

### Data Collection
- ✅ **RSS Crawler**: Fetch articles from RSS feeds (BBC, CNN, etc.)
- ✅ **Web Crawler**: Multi-threaded crawling with robots.txt support

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Run the Web App (Recommended)

```bash
python web_app/app.py
```

Visit: **http://localhost:5000**

### 3. Or Use the CLI

```bash
python main.py              # Interactive menu
python main.py --demo      # Full demonstration
```

### 4. Or Use the FastAPI Backend

```bash
python -m src.api
```

Visit Swagger UI: **http://localhost:8000/docs**

## Project Structure

```
IR Project/
├── src/                    # Core IR modules
│   ├── preprocessing.py          # Text preprocessing
│   ├── lsi_model.py              # LSI implementation
│   ├── clustering.py              # Hierarchical clustering
│   ├── search.py                  # Search engine
│   ├── dynamic_update.py          # Dynamic updates
│   ├── evaluation.py              # Metrics
│   ├── database.py                # Database layer
│   ├── crawler.py                 # Web crawler
│   ├── rss_crawler.py             # RSS feed crawler
│   └── api.py                     # FastAPI backend
│
├── web_app/                # Flask web application
│   ├── app.py             # Flask app
│   ├── templates/         # HTML templates
│   └── static/css/        # Styling
│
├── data/                   # Sample data
│   └── sample_documents.py
│
├── examples/               # Usage examples
│   ├── quick_start.py
│   └── api_usage.py
│
├── main.py                  # CLI interface
├── requirements.txt         # Dependencies
├── Dockerfile              # Container setup
├── docker-compose.yml       # Docker configuration
├── README.md                # This file
├── PROJECT_OVERVIEW.md      # Detailed specification
└── README_WEB_APP.md       # Flask app guide
```

## Usage Examples

### CLI Usage

```bash
# Interactive menu
python main.py

# Full demo
python main.py --demo

# Stemming experiment
python main.py --experiment
```

### Web App Usage

```bash
# Start Flask app
python web_app/app.py

# Access at http://localhost:5000
```

### API Usage

```python
from examples.api_usage import IRClient

client = IRClient()
client.initialize()
client.add_documents(docs)
client.build_clusters()
results = client.search("football", top_k=5)
```

### Crawl RSS Feeds

```python
from src.rss_crawler import RSSCrawler

crawler = RSSCrawler()
documents = crawler.fetch_rss_feeds([
    "http://feeds.bbci.co.uk/news/rss.xml"
])
```

## Documentation

- **README.md** (this file) - Main project overview
- **PROJECT_OVERVIEW.md** - Complete specification and features
- **README_WEB_APP.md** - Flask web app guide

## Requirements

- Python 3.8+
- NumPy, scikit-learn, NLTK
- Flask (for web app)
- FastAPI (for REST API)
- BeautifulSoup (for web crawling)

See `requirements.txt` for complete list.

## Research Features

- Compare preprocessing strategies (stemming vs. lemmatization)
- Evaluate clustering quality (Silhouette Score, Davies-Bouldin Index)
- Measure search performance (Precision@K, Recall@K)
- Dynamic document addition without full rebuild

## IR Syllabus Coverage

- **Unit 1**: Text Preprocessing (tokenization, stemming, stop-words)
- **Unit 2**: Evaluation Metrics (precision, recall, F1, MRR)
- **Unit 4**: LSI & Clustering (HAC, dendrograms, quality metrics)
- **Unit 5**: Dynamic Updates (incremental clustering, web IR)

## Technologies

- Python 3.8+
- scikit-learn (ML algorithms)
- NLTK (NLP preprocessing)
- Flask (web interface)
- FastAPI (REST API)
- BeautifulSoup (web scraping)
- Docker (containerization)

---

**Ready to use!** Start with `python web_app/app.py` or `python main.py` 🚀
