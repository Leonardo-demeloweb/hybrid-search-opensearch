"""
03_index_with_embeddings.py
===========================

Indexa documentos no OpenSearch com embeddings do Azure OpenAI.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI
from opensearchpy import OpenSearch, helpers

load_dotenv()

# Azure OpenAI
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME", "text-embedding-3-small")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

# OpenSearch
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
INDEX_NAME = "estabelecimentos_v001"

BATCH_SIZE = 20


def get_azure_client() -> AzureOpenAI:
    if not AZURE_API_KEY or not AZURE_ENDPOINT:
        raise ValueError("Configure AZURE_OPENAI_API_KEY e AZURE_OPENAI_ENDPOINT no .env")
    return AzureOpenAI(api_key=AZURE_API_KEY, api_version=AZURE_API_VERSION, azure_endpoint=AZURE_ENDPOINT)


def get_opensearch_client() -> OpenSearch:
    return OpenSearch(hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}], use_ssl=False, verify_certs=False, timeout=60)


def create_text_for_embedding(doc: dict) -> str:
    parts = [doc.get("razao_social", ""), doc.get("nome_fantasia", ""), doc.get("cnae_descricao", ""), doc.get("descricao_atividade", ""), doc.get("endereco", {}).get("cidade", ""), doc.get("endereco", {}).get("uf", "")]
    return " | ".join(filter(None, parts))


def generate_embeddings_batch(client: AzureOpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(input=texts, model=AZURE_DEPLOYMENT)
    return [item.embedding for item in response.data]


def index_documents(documents: list[dict]) -> tuple[int, int]:
    azure_client = get_azure_client()
    os_client = get_opensearch_client()
    
    print(f"📡 Azure OpenAI: {AZURE_ENDPOINT}")
    print(f"   Deployment: {AZURE_DEPLOYMENT}")
    
    info = os_client.info()
    print(f"📡 OpenSearch: {info['version']['number']}")
    
    now = datetime.now(timezone.utc).isoformat()
    total_success, total_failed = 0, 0
    
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"\n📦 Batch {batch_num}/{total_batches} ({len(batch)} docs)")
        
        texts = [create_text_for_embedding(doc) for doc in batch]
        
        print("   🧠 Gerando embeddings...")
        embeddings = generate_embeddings_batch(azure_client, texts)
        print(f"   ✅ {len(embeddings)} embeddings gerados (dim={len(embeddings[0])})")
        
        actions = []
        for doc, embedding in zip(batch, embeddings):
            doc_id = doc.pop("_id", doc.get("cnpj", "").replace(".", "").replace("/", "").replace("-", ""))
            doc["embedding"] = embedding
            doc["indexed_at"] = now
            actions.append({"_index": INDEX_NAME, "_id": doc_id, "_source": doc})
        
        print("   📥 Indexando...")
        success, failed = helpers.bulk(os_client, actions, stats_only=True, raise_on_error=False)
        total_success += success
        total_failed += failed
        print(f"   ✅ {success} indexados, ❌ {failed} falhas")
    
    os_client.indices.refresh(index=INDEX_NAME)
    return total_success, total_failed


if __name__ == "__main__":
    print("=" * 60)
    print("📥 INDEXAÇÃO COM EMBEDDINGS (Azure OpenAI)")
    print("=" * 60)
    
    data_file = Path(__file__).parent.parent / "data" / "estabelecimentos_demo.json"
    
    if not data_file.exists():
        print(f"\n❌ Arquivo não encontrado: {data_file}")
        print("   Execute primeiro: python 02_generate_data.py")
        exit(1)
    
    print(f"\n📂 Carregando: {data_file}")
    with open(data_file, "r", encoding="utf-8") as f:
        documents = json.load(f)
    print(f"   {len(documents)} documentos")
    
    try:
        success, failed = index_documents(documents)
        
        print("\n" + "=" * 60)
        print("📊 RESULTADO FINAL")
        print("=" * 60)
        print(f"   ✅ Sucesso: {success}")
        print(f"   ❌ Falhas: {failed}")
        
        os_client = get_opensearch_client()
        count = os_client.count(index=INDEX_NAME)["count"]
        print(f"\n📊 Total no índice: {count}")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        print("\nVerifique:")
        print("  1. Credenciais Azure OpenAI no .env")
        print("  2. Nome correto do deployment de embeddings")
        print("  3. OpenSearch rodando (docker-compose up)")
        exit(1)
