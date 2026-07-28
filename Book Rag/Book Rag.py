import pathlib
import chromadb
import ollama

VAULT_PATH  = "C:\\Users\\awake\\Documents\\Book Series Archive"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL  = "llama3.1"
DB_PATH     = "./book_index"
COLLECTION  = "book"


#Load
def load_notes():
    notes = []
    root = pathlib.Path(VAULT_PATH)
    for path in root.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        if text.strip():
            notes.append((str(path.relative_to(root)), text))
    return notes


# Chunk
def chunk_text(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


#Embed to Vector
def embed(text):
    return ollama.embed(model=EMBED_MODEL, input=text).embeddings[0]


# Store into ChromaDB
def get_index():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(COLLECTION)
    if collection.count() == 0:
        notes = load_notes()
        print(f"First run: indexing {len(notes)} notes (this can take a bit)...")
        for note_name, text in notes:
            for i, chunk in enumerate(chunk_text(text)):
                collection.add(
                    ids=[f"{note_name}::{i}"],       # unique id per passage
                    embeddings=[embed(chunk)],       # its vector
                    documents=[chunk],               # the original text
                    metadatas=[{"note": note_name}], # where it came from
                )
        print("Done. Index saved to", DB_PATH)
    return collection


#RaG: Retrieve and Generate
def ask(collection, question, k=5):
    hits = collection.query(query_embeddings=[embed(question)], n_results=k)
    passages = hits["documents"][0]
    sources  = [m["note"] for m in hits["metadatas"][0]]

    context = "\n\n---\n\n".join(passages)
    prompt = (
        "Answer the question using ONLY the notes below about the user's book. "
        "If the answer isn't in the notes, say you don't know.\n\n"
        f"NOTES:\n{context}\n\n"
        f"QUESTION: {question}"
    )
    answer = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    ).message.content
    return answer, sorted(set(sources))


#Index & Run
if __name__ == "__main__":
    collection = get_index()
    print("\nAsk about your book (Ctrl+C to quit):\n")
    while True:
        try:
            question = input("You: ").strip()
            if not question:
                continue
            answer, sources = ask(collection, question)
            print(f"\n{answer}\n\nsources: {', '.join(sources)}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break