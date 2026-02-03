"""
04_hybrid_queries.py
====================

Demonstra queries híbridas: Full-text + k-NN + Geo.
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from opensearchpy import OpenSearch

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

azure_client = None
os_client = None


def init_clients():
    global azure_client, os_client
    if AZURE_API_KEY and AZURE_ENDPOINT:
        azure_client = AzureOpenAI(api_key=AZURE_API_KEY, api_version=AZURE_API_VERSION, azure_endpoint=AZURE_ENDPOINT)
    os_client = OpenSearch(hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}], use_ssl=False, verify_certs=False, timeout=30)


def get_embedding(text: str) -> list[float]:
    if not azure_client:
        raise ValueError("Azure OpenAI não configurado")
    response = azure_client.embeddings.create(input=[text], model=AZURE_DEPLOYMENT)
    return response.data[0].embedding


def print_results(results: dict, title: str):
    hits = results["hits"]["hits"]
    total = results["hits"]["total"]["value"]
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")
    print(f"Total: {total} | Exibindo: {len(hits)}")
    print("-" * 60)
    
    for i, hit in enumerate(hits, 1):
        src = hit["_source"]
        score = hit.get("_score") or 0
        print(f"\n{i}. {src.get('razao_social', 'N/A')}")
        print(f"   CNAE: {src.get('cnae_codigo', 'N/A')} - {src.get('cnae_descricao', 'N/A')[:50]}...")
        print(f"   Local: {src.get('endereco', {}).get('cidade', 'N/A')}/{src.get('endereco', {}).get('uf', 'N/A')}")
        if score:
            print(f"   Score: {score:.4f}")


def query_fulltext(query: str, size: int = 5):
    """Query 1: Full-Text (BM25)"""
    body = {
        "size": size,
        "query": {"multi_match": {"query": query, "fields": ["razao_social^2", "nome_fantasia^2", "cnae_descricao^3", "descricao_atividade"], "type": "best_fields", "fuzziness": "AUTO"}},
        "_source": {"excludes": ["embedding"]}
    }
    results = os_client.search(index=INDEX_NAME, body=body)
    print_results(results, f"FULL-TEXT: '{query}'")
    return results


def query_knn(query: str, k: int = 5):
    """Query 2: k-NN (Semântica)"""
    if not azure_client:
        print("\n⚠️  k-NN requer Azure OpenAI configurado")
        return None
    
    query_vector = get_embedding(query)
    body = {
        "size": k,
        "query": {"knn": {"embedding": {"vector": query_vector, "k": k}}},
        "_source": {"excludes": ["embedding"]}
    }
    results = os_client.search(index=INDEX_NAME, body=body)
    print_results(results, f"k-NN SEMÂNTICA: '{query}'")
    return results


def query_geo(lat: float, lon: float, distance: str = "50km", size: int = 5):
    """Query 3: Geoespacial"""
    body = {
        "size": size,
        "query": {"bool": {"filter": {"geo_distance": {"distance": distance, "localizacao": {"lat": lat, "lon": lon}}}}},
        "sort": [{"_geo_distance": {"localizacao": {"lat": lat, "lon": lon}, "order": "asc", "unit": "km"}}],
        "_source": {"excludes": ["embedding"]}
    }
    results = os_client.search(index=INDEX_NAME, body=body)
    
    for hit in results["hits"]["hits"]:
        if "sort" in hit:
            hit["_source"]["_distancia_km"] = round(hit["sort"][0], 2)
    
    print_results(results, f"GEO: {distance} de ({lat}, {lon})")
    print("\n📍 Distâncias:")
    for hit in results["hits"]["hits"]:
        dist = hit["_source"].get("_distancia_km", "N/A")
        print(f"   {hit['_source']['razao_social'][:40]}: {dist} km")
    return results


def query_hybrid(text_query: str, lat: float, lon: float, distance: str = "100km", size: int = 5, use_knn: bool = True):
    """Query 4: HÍBRIDA (Full-text + k-NN + Geo)"""
    must = [{"multi_match": {"query": text_query, "fields": ["razao_social^2", "nome_fantasia^2", "cnae_descricao^3", "descricao_atividade"], "type": "best_fields", "fuzziness": "AUTO"}}]
    should = []
    filter_clauses = [{"geo_distance": {"distance": distance, "localizacao": {"lat": lat, "lon": lon}}}]
    
    if use_knn and azure_client:
        query_vector = get_embedding(text_query)
        should.append({"knn": {"embedding": {"vector": query_vector, "k": size * 2}}})
    
    body = {
        "size": size,
        "query": {"bool": {"must": must, "should": should, "filter": filter_clauses}},
        "sort": ["_score", {"_geo_distance": {"localizacao": {"lat": lat, "lon": lon}, "order": "asc", "unit": "km"}}],
        "_source": {"excludes": ["embedding"]}
    }
    
    results = os_client.search(index=INDEX_NAME, body=body)
    
    for hit in results["hits"]["hits"]:
        if "sort" in hit and len(hit["sort"]) > 1:
            hit["_source"]["_distancia_km"] = round(hit["sort"][1], 2)
    
    knn_label = "+ k-NN" if use_knn and azure_client else "(sem k-NN)"
    print_results(results, f"HÍBRIDA {knn_label}: '{text_query}' em {distance} de ({lat}, {lon})")
    return results


def run_demo():
    print("\n" + "=" * 60)
    print("🚀 DEMONSTRAÇÃO DE QUERIES HÍBRIDAS")
    print("=" * 60)
    
    if not os_client.indices.exists(index=INDEX_NAME):
        print(f"\n❌ Índice '{INDEX_NAME}' não existe.")
        print("   Execute: python 01_create_index.py")
        return
    
    count = os_client.count(index=INDEX_NAME)["count"]
    print(f"\n📊 Documentos no índice: {count}")
    
    if count == 0:
        print("   ⚠️  Índice vazio. Execute:")
        print("   python 02_generate_data.py")
        print("   python 03_index_with_embeddings.py")
        return
    
    SP_LAT, SP_LON = -23.5505, -46.6333
    
    print("\n" + "-" * 60)
    print("📋 EXEMPLOS DE QUERIES")
    print("-" * 60)
    
    query_fulltext("granito mármore pedra", size=3)
    
    if azure_client:
        query_knn("materiais para construção de estradas", k=3)
    else:
        print("\n⚠️  Pulando k-NN (Azure OpenAI não configurado)")
    
    query_geo(SP_LAT, SP_LON, "30km", size=3)
    
    query_hybrid(text_query="areia cascalho construção", lat=SP_LAT, lon=SP_LON, distance="50km", size=3)
    
    print("\n" + "=" * 60)
    print("✅ Demonstração concluída!")
    print("=" * 60)


if __name__ == "__main__":
    init_clients()
    
    print("=" * 60)
    print("🔍 QUERIES HÍBRIDAS - OpenSearch")
    print("=" * 60)
    print(f"\n📡 OpenSearch: {OPENSEARCH_HOST}:{OPENSEARCH_PORT}")
    
    if azure_client:
        print(f"📡 Azure OpenAI: {AZURE_ENDPOINT}")
        print(f"   Deployment: {AZURE_DEPLOYMENT}")
    else:
        print("⚠️  Azure OpenAI: Não configurado (k-NN desabilitado)")
    
    run_demo()
