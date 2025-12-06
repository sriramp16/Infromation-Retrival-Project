from src.preprocessing import DocumentPreprocessor
from src.lsi_model import LSIModel
from data.sample_documents import get_sample_documents


def main():
    docs = get_sample_documents()
    pre = DocumentPreprocessor()
    processed = pre.preprocess_documents(docs)

    lsi = LSIModel(n_topics=5)
    lsi.fit(processed)

    topics = lsi.get_top_terms_per_topic(n_terms=12)

    for tidx, terms in enumerate(topics):
        print(f"\nTopic {tidx}:")
        for term, weight in sorted(terms, key=lambda x: -abs(x[1])):
            print(f"  {term}: {weight:.4f}")

    print("\n=== Topic 0 (top terms) ===")
    for term, weight in sorted(topics[0], key=lambda x: -abs(x[1]))[:12]:
        print(f"{term}: {weight:.4f}")


if __name__ == '__main__':
    main()
