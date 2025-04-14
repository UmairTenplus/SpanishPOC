import os
import uuid
import openai
from dotenv import load_dotenv
from pinecone import Pinecone
from tqdm import tqdm

# === Load environment variables ===
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# === Connect to Pinecone ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "spanishpoc"
index = pc.Index(index_name)

# === Load chunked text files ===
chunk_dir = "chunked_output"
all_chunks = []

for fname in os.listdir(chunk_dir):
    if fname.endswith(".txt"):
        file_path = os.path.join(chunk_dir, fname)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                file_basename = fname.split("_chunk_")[0]
                chunk_number = fname.split("_chunk_")[-1].replace(".txt", "")
                chunk_id = str(uuid.uuid4())

                all_chunks.append({
                    "id": chunk_id,
                    "text": content,
                    "metadata": {
                        "source_file": file_basename,
                        "chunk_number": chunk_number
                    }
                })

print(f"\n📄 Loaded {len(all_chunks)} chunks from '{chunk_dir}' for embedding and upload.\n")

# === Function to embed text ===
def get_embeddings(texts):
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [d["embedding"] for d in response["data"]]

# === Upload to Pinecone in batches with per-file logs ===
batch_size = 100
for i in tqdm(range(0, len(all_chunks), batch_size), desc="Uploading batches"):
    batch = all_chunks[i:i + batch_size]
    texts = [item["text"] for item in batch]
    embeddings = get_embeddings(texts)

    to_upsert = []
    for j, item in enumerate(batch):
        vector = {
            "id": item["id"],
            "values": embeddings[j],
            "metadata": item["metadata"]
        }
        to_upsert.append(vector)

        # 🧾 Detailed log for each chunk
        print(f"✅ Embedded & ready to upload: File = '{item['metadata']['source_file']}', Chunk = {item['metadata']['chunk_number']}")

    index.upsert(vectors=to_upsert)

print("\n✅ All chunks embedded and uploaded to Pinecone successfully.")
