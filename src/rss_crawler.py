

import feedparser
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import time
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RSSCrawler:

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.timeout = 10
    
    def fetch_rss_feeds(self, feed_urls: List[str], max_articles_per_feed: int = 20) -> List[Dict]:
        documents = []
        doc_id = 1
        
        for url in feed_urls:
            try:
                logger.info(f"Fetching feed: {url}")
                feed = feedparser.parse(url)
                
                count = 0
                for entry in feed.entries:
                    if count >= max_articles_per_feed:
                        break
                    
                    title = entry.get('title', 'No title')
                    summary = entry.get('summary', entry.get('description', ''))
                    link = entry.get('link', '')
                    published = entry.get('published', '')
                    
                    try:
                        content = self.fetch_full_content(link) if link else summary
                    except Exception as e:
                        logger.warning(f"Could not fetch full content from {link}: {e}")
                        content = summary
                    
                    if content:
                        doc = {
                            'id': doc_id,
                            'title': title,
                            'content': content,
                            'source': 'RSS Feed',
                            'url': link,
                            'published': published,
                            'category': 'News'
                        }
                        documents.append(doc)
                        doc_id += 1
                        count += 1
                
                logger.info(f"Fetched {count} articles from {url}")
                
            except Exception as e:
                logger.error(f"Error fetching feed {url}: {e}")
                continue
        
        return documents
    
    def fetch_full_content(self, url: str) -> str:
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(['script', 'style']):
                script.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
            
            return text[:2000]
            
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return ""
    
    def save_documents(self, documents: List[Dict], filename: str) -> None:
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(documents)} documents to {filename}")
        except Exception as e:
            logger.error(f"Error saving documents: {e}")


def main() -> List[Dict]:
    rss_feeds = [
        "http://feeds.bbci.co.uk/news/rss.xml",
        "http://rss.cnn.com/rss/edition.rss",
        "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "https://www.theguardian.com/world/rss",
    ]
    
    crawler = RSSCrawler()
    
    print("\n" + "=" * 60)
    print("RSS Feed Crawler")
    print("=" * 60)
    print(f"\nFetching articles from {len(rss_feeds)} feeds...")
    print("(Max 20 articles per feed)\n")
    
    documents = crawler.fetch_rss_feeds(rss_feeds, max_articles_per_feed=20)
    
    crawler.save_documents(documents, "data/crawled_news.json")
    
    print("\n" + "=" * 60)
    print(f"✓ Successfully crawled {len(documents)} articles")
    print("=" * 60)
    
    print("\nSample documents:")
    for doc in documents[:3]:
        print(f"\n- {doc['title'][:60]}")
        print(f"  Source: {doc['source']}")
        print(f"  Content: {doc['content'][:100]}...")
    
    return documents


if __name__ == "__main__":
    documents = main()