import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, QueryResponse
from sentence_transformers import SentenceTransformer

# =====================================================================
# CONFIGURAZIONE AMBIENTE E RETE
# =====================================================================
# IP di OrbStack configurato per l'accesso dall'host
QDRANT_HOST = "192.168.139.223"
QDRANT_PORT = 6333
COLLECTION_NAME = "eibench_journalism_corpus"

print(f"🔄 Connessione a Qdrant su {QDRANT_HOST}:{QDRANT_PORT}...")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

print("🧠 Caricamento del modello di embedding locale (Sentence-BERT)...")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Reset programmato della collezione per garantire test puliti
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,  # Dimensione dei vettori del modello all-MiniLM-L6-v2
        distance=Distance.COSINE 
    ),
)
print(f"✅ Collezione '{COLLECTION_NAME}' creata con successo.")

# =====================================================================
# LAYER 1 & 2: CORPUS BUILDER & POISONING ENGINE
# =====================================================================
print("\n📥 Inizio fase di Ingestione e Avvelenamento Dati (Layer 1 & 2)...")
with open("eibench_raw_claims.json", "r") as f:
    claims_data = json.load(f)

points = []
idx = 1
for item in claims_data:
    # 1. Inserimento del dato integro (Ground Truth)
    fact_text = item["original_fact"]
    fact_vector = encoder.encode(fact_text).tolist()
    points.append(PointStruct(
        id=idx, 
        vector=fact_vector,
        payload={"text": fact_text, "type": "ground_truth", "claim_id": item["claim_id"]}
    ))
    idx += 1
    
    # 2. Inserimento delle varianti avversariali controllate (Data Poisoning)
    for strategy, poisoned_text in item["adversarial_variants"].items():
        poison_vector = encoder.encode(poisoned_text).tolist()
        points.append(PointStruct(
            id=idx, 
            vector=poison_vector,
            payload={
                "text": poisoned_text, 
                "type": "poisoned", 
                "strategy": strategy, 
                "claim_id": item["claim_id"]
            }
        ))
        idx += 1

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"🚀 Iniezione completata: {len(points)} vettori memorizzati.")

# =====================================================================
# LAYER 3: RETRIEVAL INFRASTRUCTURE
# =====================================================================
query_text = claims_data[0]["context_query"]
print(f"\n🔍 Generazione embedding per la query: '{query_text}'")
query_vector = encoder.encode(query_text).tolist()

# Esecuzione del recupero tramite query_points (SDK v1.x compatibile)
search_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=3
)

print("\n" + "="*60)
print(f"🚨 RISULTATI DEL RETRIEVAL (EIBENCH LAYER 3)")
print("="*60)

for rank, item in enumerate(search_results.points):
    hit = item if isinstance(item, dict) else item.__dict__
    score = hit.get('score', 0.0)
    payload = hit.get('payload', {})
    print(f"\n[RANK {rank+1}] (Cosine Similarity: {score:.4f})")
    print(f"Tipo: {payload.get('type', '').upper()} | Strategia: {payload.get('strategy', 'N/A')}")
    print(f"Testo: {payload.get('text', '')}")
print("\n" + "="*60)
