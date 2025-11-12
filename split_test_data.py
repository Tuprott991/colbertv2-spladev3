"""
Split test.tsv into separate files for ColBERTv2 evaluation.

Your test.tsv format: query \t passage
This script will create:
1. collection.tsv - All unique passages with IDs (pid \t passage_text)
2. queries.tsv - All unique queries with IDs (qid \t query_text)
3. qrels.tsv - Query-passage relevance pairs (qid \t 0 \t pid \t 1)
"""

import csv
import os
from collections import defaultdict

def split_test_file(input_file, output_dir):
    """
    Split test.tsv into collection, queries, and qrels files.
    Assumes format: query \t passage
    """
    print(f"Processing {input_file}...")
    print("=" * 80)
    
    # Store unique queries and passages
    queries = {}  # query_text -> qid
    passages = {}  # passage_text -> pid
    qrels_data = []  # (qid, pid) pairs
    
    query_counter = 0
    passage_counter = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, delimiter='\t')
            
            for line_num, row in enumerate(reader, 1):
                if len(row) < 2:
                    print(f"Warning: Skipping line {line_num} - insufficient columns")
                    continue
                
                query_text = row[0].strip()
                passage_text = row[1].strip()
                
                if not query_text or not passage_text:
                    continue
                
                # Add query if new
                if query_text not in queries:
                    queries[query_text] = query_counter
                    query_counter += 1
                
                # Add passage if new
                if passage_text not in passages:
                    passages[passage_text] = passage_counter
                    passage_counter += 1
                
                # Record relevance pair
                qid = queries[query_text]
                pid = passages[passage_text]
                qrels_data.append((qid, pid))
                
                if line_num % 1000 == 0:
                    print(f"Processed {line_num} lines... "
                          f"({query_counter} queries, {passage_counter} passages)")
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    print("=" * 80)
    print(f"Total lines processed: {line_num}")
    print(f"Unique queries: {query_counter}")
    print(f"Unique passages: {passage_counter}")
    print(f"Query-passage pairs: {len(qrels_data)}")
    print("=" * 80)
    
    # Write collection.tsv
    collection_file = os.path.join(output_dir, "collection.tsv")
    print(f"\nWriting {collection_file}...")
    with open(collection_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # Sort by pid to write in order
        sorted_passages = sorted(passages.items(), key=lambda x: x[1])
        for passage_text, pid in sorted_passages:
            writer.writerow([pid, passage_text])
    print(f"✓ Collection saved: {passage_counter} passages")
    
    # Write queries.tsv
    queries_file = os.path.join(output_dir, "queries.dev.tsv")
    print(f"\nWriting {queries_file}...")
    with open(queries_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # Sort by qid to write in order
        sorted_queries = sorted(queries.items(), key=lambda x: x[1])
        for query_text, qid in sorted_queries:
            writer.writerow([qid, query_text])
    print(f"✓ Queries saved: {query_counter} queries")
    
    # Write qrels.tsv
    qrels_file = os.path.join(output_dir, "qrels.dev.tsv")
    print(f"\nWriting {qrels_file}...")
    with open(qrels_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for qid, pid in qrels_data:
            # Format: qid \t 0 \t pid \t relevance (1 = relevant)
            writer.writerow([qid, 0, pid, 1])
    print(f"✓ Qrels saved: {len(qrels_data)} relevance pairs")
    
    print("\n" + "=" * 80)
    print("SUCCESS! Files created:")
    print(f"  • {collection_file}")
    print(f"  • {queries_file}")
    print(f"  • {qrels_file}")
    print("=" * 80)
    print("\nYou can now run colbertv2.py for evaluation!")

def main():
    base_dir = r"d:\Github Repos\ColBERT-and-SPLADE"
    test_file = os.path.join(base_dir, "test.tsv")
    
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found!")
        return
    
    print("ColBERTv2 Test Data Splitter")
    print("=" * 80)
    print(f"Input: {test_file}")
    print(f"Format: query \\t passage")
    print("=" * 80)
    
    split_test_file(test_file, base_dir)

if __name__ == "__main__":
    main()
