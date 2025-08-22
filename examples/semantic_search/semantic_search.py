import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Tuple

from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, db_path: str = "semsearch.db", model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model_name = model_name
        self.model = None
        self.conn = None

    def _get_model(self):
        """Lazy load the sentence transformer model"""
        if self.model is None:
            print(f"Loading model {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def _get_connection(self):
        """Get database connection, load SQLite Vector extension
        and ensure schema is created"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)

            self.conn.enable_load_extension(True)
            self.conn.load_extension("./vector.so")
            self.conn.enable_load_extension(False)

            # Check if sqlite-vector is available
            try:
                self.conn.execute("SELECT vector_version()")
            except sqlite3.OperationalError:
                print("Error: sqlite-vector extension not found.")
                print(
                    "Download it from https://github.com/sqliteai/sqlite-vector/releases")
                sys.exit(1)

            self._create_schema()
        return self.conn

    def _create_schema(self):
        """Create the documents table with vector support"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create vector table using sqlite-vector extension
        # The default model 'all-MiniLM-L6-v2' produces 384-dimensional embeddings

        # Initialize the vector
        cursor.execute("""
            SELECT vector_init('documents', 'embedding', 'type=FLOAT32,dimension=384');
        """)

        conn.commit()

    def _chunk_text(self, text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better semantic search"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)

        # Return original if no chunks created
        return chunks if chunks else [text]

    def index_file(self, filepath: str) -> int:
        """Index a single file and return number of chunks processed"""
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return 0

        model = self._get_model()
        conn = self._get_connection()

        cursor = conn.execute(
            "SELECT id FROM documents WHERE filepath = ?", (filepath,))
        if cursor.fetchone() is not None:
            print(f"File already indexed: {filepath}")
            return 0

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return 0

        if not content:
            print(f"Empty file: {filepath}")
            return 0

        cursor = conn.cursor()

        # Split content into chunks.
        # The default model truncates text after 256 word pieces
        chunks = self._chunk_text(content)
        chunk_count = 0

        for chunk in chunks:
            # Generate embedding and insert into database
            embedding = model.encode(chunk)
            embedding_json = json.dumps(embedding.tolist())

            cursor.execute(
                "INSERT INTO documents (filepath, content, embedding) VALUES (?, ?, vector_as_f32(?))",
                (filepath, chunk, embedding_json)
            )
            chunk_count += 1

        conn.commit()

        # Perform quantization on the vector column
        cursor.execute("""
            SELECT vector_quantize('documents', 'embedding');
        """)

        print(f"Indexed {filepath}: {chunk_count} chunks")
        return chunk_count

    def index_directory(self, directory: str) -> int:
        """Index all text files in a directory"""
        total_chunks = 0
        text_extensions = {'.txt', '.md', '.mdx', '.py', '.js',
                           '.html', '.css', '.sql', '.json', '.xml'}

        for root, _, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in text_extensions:
                    filepath = os.path.join(root, file)
                    total_chunks += self.index_file(filepath)

        return total_chunks

    def search(self, query: str, limit: int = 3) -> Tuple[float, List[Tuple[str, str, float]]]:
        """Search for similar documents"""
        model = self._get_model()
        conn = self._get_connection()

        # Generate query embedding
        query_embedding = model.encode(query)
        query_json = json.dumps(query_embedding.tolist())

        # Search using sqlite-vec cosine similarity
        cursor = conn.cursor()
        start_time = time.time()
        cursor.execute("""
            SELECT d.id, d.filepath, d.content, v.distance
            FROM documents AS d
                JOIN vector_quantize_scan('documents', 'embedding', vector_as_f32(?), ?) AS v
                ON d.id = v.rowid;
        """, (query_json, limit))
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        results = []
        for id, filepath, content, distance in cursor.fetchall():
            results.append((filepath, content, distance))

        return (elapsed_ms, results)

    def stats(self):
        """Print database statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT filepath) FROM documents")
        file_count = cursor.fetchone()[0]

        print(f"Database: {self.db_path}")
        print(f"Files indexed: {file_count}")
        print(f"Document chunks: {doc_count}")

    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("Database connection closed.")
