"""
01_create_index.py
==================

Cria o índice híbrido para busca de estabelecimentos.
Suporta: Full-text (BM25) + Vetorial (k-NN) + Geoespacial

Compatível com OpenSearch 2.x e Elasticsearch 8.x
"""

import json
from opensearchpy import OpenSearch

# =============================================================================
# Configuração
# =============================================================================

# OpenSearch local (docker-compose)
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
OPENSEARCH_USER = None  # None se DISABLE_SECURITY_PLUGIN=true
OPENSEARCH_PASS = None

# Índice
INDEX_NAME = "estabelecimentos_v001"

# =============================================================================
# Cliente
# =============================================================================

def get_client() -> OpenSearch:
    """Cria cliente OpenSearch."""
    auth = (OPENSEARCH_USER, OPENSEARCH_PASS) if OPENSEARCH_USER else None
    
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=auth,
        use_ssl=False,
        verify_certs=False,
        timeout=30,
    )

# =============================================================================
# Mapping do Índice Híbrido
# =============================================================================

INDEX_SETTINGS = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "knn": True,  # Habilita k-NN
            "knn.algo_param.ef_search": 100  # Parâmetro de busca HNSW
        },
        "analysis": {
            "filter": {
                "brazilian_stop": {
                    "type": "stop",
                    "stopwords": "_brazilian_"
                },
                "brazilian_stemmer": {
                    "type": "stemmer",
                    "language": "brazilian"
                },
                "brazilian_keywords": {
                    "type": "keyword_marker",
                    "keywords": ["cnpj", "cnae", "mei", "ltda", "eireli"]
                }
            },
            "analyzer": {
                "brazilian_text": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "asciifolding",
                        "brazilian_stop",
                        "brazilian_keywords",
                        "brazilian_stemmer"
                    ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            # =====================================================
            # Identificação
            # =====================================================
            "cnpj": {
                "type": "keyword",
                "doc_values": True
            },
            "razao_social": {
                "type": "text",
                "analyzer": "brazilian_text",
                "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 256}
                }
            },
            
            # =====================================================
            # Classificação CNAE
            # =====================================================
            "cnae_codigo": {
                "type": "keyword"
            },
            "cnae_secao": {
                "type": "keyword"
            },
            "cnae_descricao": {
                "type": "text",
                "analyzer": "brazilian_text"
            },
            
            # =====================================================
            # Texto para Busca Full-Text (BM25)
            # =====================================================
            "atividade_descritiva": {
                "type": "text",
                "analyzer": "brazilian_text"
            },
            "texto_busca": {
                "type": "text",
                "analyzer": "brazilian_text"
            },
            
            # =====================================================
            # Vetor para Busca Semântica (k-NN)
            # =====================================================
            "embedding": {
                "type": "knn_vector",
                "dimension": 1536,  # text-embedding-3-small
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",  # nmslib suporta >1024 dims
                    "parameters": {
                        "ef_construction": 128,
                        "m": 16
                    }
                }
            },
            
            # =====================================================
            # Geolocalização
            # =====================================================
            "localizacao": {
                "type": "geo_point"
            },
            "endereco": {
                "type": "object",
                "properties": {
                    "logradouro": {"type": "text"},
                    "numero": {"type": "keyword"},
                    "complemento": {"type": "text"},
                    "bairro": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "municipio": {"type": "keyword"},
                    "uf": {"type": "keyword"},
                    "cep": {"type": "keyword"}
                }
            },
            
            # =====================================================
            # Metadados
            # =====================================================
            "situacao_cadastral": {
                "type": "keyword"
            },
            "porte": {
                "type": "keyword"
            },
            "natureza_juridica": {
                "type": "keyword"
            },
            "data_abertura": {
                "type": "date",
                "format": "yyyy-MM-dd"
            },
            "capital_social": {
                "type": "float"
            },
            
            # =====================================================
            # Controle
            # =====================================================
            "indexed_at": {
                "type": "date"
            }
        }
    }
}

# =============================================================================
# Funções
# =============================================================================

def create_index(client: OpenSearch, delete_if_exists: bool = False) -> dict:
    """
    Cria o índice híbrido.
    
    Args:
        client: Cliente OpenSearch
        delete_if_exists: Se True, deleta índice existente
        
    Returns:
        Resposta da criação
    """
    # Verifica se existe
    if client.indices.exists(index=INDEX_NAME):
        if delete_if_exists:
            print(f"🗑️  Deletando índice existente: {INDEX_NAME}")
            client.indices.delete(index=INDEX_NAME)
        else:
            print(f"⚠️  Índice já existe: {INDEX_NAME}")
            return {"acknowledged": False, "message": "Index already exists"}
    
    # Cria índice
    print(f"📦 Criando índice: {INDEX_NAME}")
    response = client.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)
    
    print(f"✅ Índice criado com sucesso!")
    print(f"   - Shards: {INDEX_SETTINGS['settings']['index']['number_of_shards']}")
    print(f"   - k-NN: Habilitado (HNSW, dim=1536)")
    print(f"   - Analyzer: brazilian_text")
    
    return response


def get_mapping(client: OpenSearch) -> dict:
    """Retorna o mapping atual do índice."""
    return client.indices.get_mapping(index=INDEX_NAME)


def get_settings(client: OpenSearch) -> dict:
    """Retorna as settings atuais do índice."""
    return client.indices.get_settings(index=INDEX_NAME)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 CRIAÇÃO DE ÍNDICE HÍBRIDO - OpenSearch")
    print("=" * 60)
    
    # Conectar
    client = get_client()
    
    # Verificar conexão
    info = client.info()
    print(f"\n📡 Conectado ao OpenSearch {info['version']['number']}")
    print(f"   Cluster: {info['cluster_name']}")
    
    # Criar índice
    print()
    result = create_index(client, delete_if_exists=True)
    
    # Mostrar mapping
    if result.get("acknowledged"):
        print("\n📋 Mapping criado:")
        mapping = get_mapping(client)
        print(json.dumps(mapping, indent=2, default=str)[:500] + "...")
    
    print("\n" + "=" * 60)
    print("✅ Pronto para indexar documentos!")
    print("=" * 60)
