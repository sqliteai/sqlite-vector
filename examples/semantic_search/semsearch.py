#!/usr/bin/env python3
"""
Semantic Search CLI Tool using SQLite + sqlite-vec + sentence-transformers
Usage:
  semsearch "query text"              # Search for similar documents
  semsearch -i /path/to/documents     # Index documents from directory
  semsearch -i /path/to/file.txt      # Index single file
"""

import argparse
import os
import sys

from semantic_search import SemanticSearch


def main():
    parser = argparse.ArgumentParser(
        description="Semantic search using SQLite + sqlite-vector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  semsearch "machine learning algorithms"
  semsearch -i /path/to/documents
  semsearch -i document.txt
  semsearch --stats
        """
    )

    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("-i", "--index", metavar="PATH",
                        help="Index file or directory")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of results to return (default: 5)")
    parser.add_argument("--db", default="semsearch.db",
                       help="Database file path (default: semsearch.db)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence transformer model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--stats", action="store_true",
                        help="Show database statistics")
    parser.add_argument("--repl", action="store_true",
                        help="Run in interactive (keep model in memory)")

    args = parser.parse_args()

    if not any([args.query, args.index, args.stats, args.repl]):
        parser.print_help()
        return

    searcher = SemanticSearch(args.db, args.model)

    try:
        if args.stats:
            searcher.stats()

        elif args.index:
            if os.path.isdir(args.index):
                total = searcher.index_directory(args.index)
                print(f"Total chunks indexed: {total}")
            else:
                searcher.index_file(args.index)

        elif args.query:
            elapsed_ms, results = searcher.search(args.query, args.limit)

            if not results:
                print("No results found.")
                return

            print(f"Results for: '{args.query}' in {elapsed_ms}ms\n")
            for i, (filepath, content, similarity) in enumerate(results, 1):
                print(f"{i}. {filepath} (similarity: {similarity:.3f})")
                # Show first 200 chars of content
                preview = content[:200] + \
                    "..." if len(content) > 200 else content
                print(f"   {preview}\n")

        if args.repl:
            print("Entering interactive mode (keep the model in memory).\nType 'help' for commands, 'exit' to quit.")
            while True:
                try:
                    cmd = input("semsearch> ").strip()
                    if not cmd:
                        continue
                    if cmd in {"exit", "quit"}:
                        break
                    if cmd == "help":
                        print(
                            "Commands: search <query>, index <file|dir>, stats, exit")
                        continue
                    if cmd.startswith("search "):
                        query = cmd[len("search "):].strip()
                        elapsed_ms, results = searcher.search(
                            query, args.limit)
                        if not results:
                            print("No results found.")
                            continue
                        print(f"Results for: '{query}' in {elapsed_ms}ms\n")
                        for i, (filepath, content, similarity) in enumerate(results, 1):
                            print(
                                f"{i}. {filepath} (similarity: {similarity:.3f})")
                            preview = content[:200] + \
                                ("..." if len(content) > 200 else "")
                            print(f"   {preview}\n")
                        continue
                    if cmd.startswith("index "):
                        path = cmd[len("index "):].strip()
                        if os.path.isdir(path):
                            total = searcher.index_directory(path)
                            print(f"Total chunks indexed: {total}")
                        else:
                            searcher.index_file(path)
                        continue
                    if cmd == "stats":
                        searcher.stats()
                        continue
                    print("Unknown command. Type 'help' for available commands.")
                except KeyboardInterrupt:
                    print("\nExiting REPL.")
                    break
                except Exception as e:
                    print(f"Error: {e}")

            if searcher:
                searcher.close()
            return

    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if searcher:
            searcher.close()


if __name__ == "__main__":
    main()
