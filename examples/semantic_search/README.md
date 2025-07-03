# Semantic Search Example with sqlite-vector

This example in Python demonstrates how to build a semantic search engine using the [sqlite-vector](https://github.com/sqliteai/sqlite-vector) extension and a Sentence Transformer model. It allows you to index and search documents using vector similarity, powered by a local LLM embedding model.

### How it works

- **Embeddings**: Uses [sentence-transformers](https://huggingface.co/sentence-transformers) to generate dense vector representations (embeddings) for text. The default model is [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a fast, lightweight model (384 dimensions) suitable for semantic search and retrieval tasks.
- **Vector Store and Search**: Embeddings are stored in SQLite using the [`sqlite-vector`](https://github.com/sqliteai/sqlite-vector) extension, enabling fast similarity search (cosine distance) directly in the database.
- **Sample Data**: The `samples/` directory contains example documents you can index and search immediately.

### Installation

1. Download the `sqlite-vector` extension for your platform [here](https://github.com/sqliteai/sqlite-vector/releases).

2. Extract the `vector.so` file in the main directory of the project.

3. Install the dependencies:


```bash
$ python -m venv venv

$ source venv/bin/activate

$ pip install -r requirements.txt
```

4. On first use, the required model will be downloaded automatically.

### Usage

Use the interactive mode to keep the model in memory and run multiple queries efficiently:

```bash
python semsearch.py --repl

# Index a directory of documents
semsearch> index ./samples

# Search for similar documents
semsearch> search "neural network architectures for image recognition"
```

### Example Queries

Try these queries to test semantic similarity:

- "neural network architectures for image recognition"
- "reinforcement learning in autonomous systems"
- "explainable artificial intelligence methods"
- "AI governance and regulatory compliance"
- "network intrusion detection systems"

**Note:**
- Supported extension are `.md`, `.txt`, `.py`, `.js`, `.html`, `.css`, `.sql`, `.json`, `.xml`.
- For more details, see the code in `semsearch.py` and `semantic_search.py`.