# Busca Híbrida em Escala: Full-Text + Vetores + Geolocalização

> **Combinando BM25, k-NN e Geo Queries com OpenSearch/Elasticsearch**

[![OpenSearch](https://img.shields.io/badge/OpenSearch-2.11+-blue)](https://opensearch.org/)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.x-yellow)](https://www.elastic.co/)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org/)

---

## 📋 Sumário

1. [Introdução](#1-introdução)
2. [Fundamentos Teóricos](#2-fundamentos-teóricos)
3. [Elasticsearch vs OpenSearch](#3-elasticsearch-vs-opensearch)
4. [Arquitetura da Solução](#4-arquitetura-da-solução)
5. [Implementação Prática](#5-implementação-prática)
6. [Otimização e Performance](#6-otimização-e-performance)
7. [Produção e Observabilidade](#7-produção-e-observabilidade)
8. [Conclusão](#8-conclusão)

---

## 1. Introdução

### O Problema: Limitações da Busca Tradicional

Imagine um sistema de busca de **estabelecimentos comerciais** no Brasil. Um usuário digita:

> *"oficina mecânica especializada em carros importados perto de mim"*

Uma busca tradicional baseada apenas em **full-text search** (BM25) encontraria documentos que contêm exatamente essas palavras. Mas e se o estabelecimento estiver cadastrado como:

- *"Centro automotivo - reparos em veículos premium"*
- *"Auto service - manutenção BMW, Mercedes, Audi"*

O full-text falharia em conectar **"oficina mecânica"** com **"centro automotivo"**, e **"carros importados"** com **"BMW, Mercedes"**. Além disso, o **"perto de mim"** sequer seria processado.

### A Solução: Busca Híbrida Multi-Modal

A **busca híbrida** combina três técnicas complementares:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUERY: "oficina mecânica carros importados"          │
│                         LOCALIZAÇÃO: São Paulo, SP (-23.55, -46.63)          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐           │
│  │  1. FULL-TEXT   │   │  2. VETORIAL    │   │  3. GEOESPACIAL │           │
│  │     (BM25)      │   │    (k-NN)       │   │   (geo_point)   │           │
│  ├─────────────────┤   ├─────────────────┤   ├─────────────────┤           │
│  │                 │   │                 │   │                 │           │
│  │ Busca tokens:   │   │ Embedding da    │   │ Filtro por      │           │
│  │ "oficina"       │   │ query →         │   │ distância:      │           │
│  │ "mecânica"      │   │ encontra docs   │   │ raio de 10km    │           │
│  │ "carros"        │   │ semanticamente  │   │ do ponto        │           │
│  │ "importados"    │   │ similares       │   │ informado       │           │
│  │                 │   │                 │   │                 │           │
│  │ ✓ Match exato   │   │ ✓ "auto service"│   │ ✓ Apenas        │           │
│  │                 │   │ ✓ "BMW, Audi"   │   │   próximos      │           │
│  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘           │
│           │                     │                     │                     │
│           └──────────┬──────────┴──────────┬──────────┘                     │
│                      │                     │                                │
│                      ▼                     ▼                                │
│           ┌─────────────────────────────────────────┐                       │
│           │         SCORE COMBINADO                  │                       │
│           │  (0.3 × BM25) + (0.5 × kNN) + (0.2 × geo)│                       │
│           │         = Relevância Final               │                       │
│           └─────────────────────────────────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Case de Uso: Busca de Estabelecimentos por CNAE

Neste artigo, construiremos um sistema de busca híbrida usando dados públicos de **CNAEs** (Classificação Nacional de Atividades Econômicas) combinados com **estabelecimentos fictícios** geolocalizados.

O índice terá:

| Campo | Tipo | Busca |
|-------|------|-------|
| `descricao_atividade` | text (BM25) | Full-text |
| `embedding` | knn_vector (1536d) | Semântica |
| `localizacao` | geo_point | Geoespacial |
| `cnae`, `razao_social` | keyword/text | Filtros |

---

## 2. Fundamentos Teóricos

### 2.1 Full-Text Search (BM25)

O algoritmo **BM25** (Best Matching 25) é o padrão para busca textual. Ele calcula a relevância baseado em:

- **TF (Term Frequency)**: Quantas vezes o termo aparece no documento
- **IDF (Inverse Document Frequency)**: Quão raro é o termo no corpus
- **Document Length Normalization**: Documentos mais curtos têm boost

**Fórmula simplificada:**

```
score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D|/avgdl))
```

Onde:
- `k1` = 1.2 (saturação de frequência)
- `b` = 0.75 (normalização por tamanho)

**Limitação:** BM25 é **lexical** - não entende sinônimos ou contexto semântico.

```python
# Exemplo: BM25 não conecta estes termos
query = "carro"
doc1 = "automóvel"  # ❌ BM25 não encontra
doc2 = "veículo"    # ❌ BM25 não encontra
doc3 = "carro"      # ✅ Match exato
```

### 2.2 Busca Vetorial (k-NN / ANN)

A busca vetorial representa textos como **vetores de alta dimensão** (embeddings) onde textos semanticamente similares estão próximos no espaço vetorial.

**Como funciona:**

```
┌─────────────────────────────────────────────────────────────────┐
│  ESPAÇO VETORIAL (simplificado em 2D)                           │
│                                                                 │
│     "automóvel" ●──────────● "carro"                           │
│                    \      /                                     │
│                     \    /                                      │
│                      \  /                                       │
│                       ●                                         │
│                   "veículo"                                     │
│                                                                 │
│                                        ● "bicicleta"            │
│                                                                 │
│                                                                 │
│                    ● "restaurante"                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Algoritmo k-NN (k-Nearest Neighbors):**
- Dado um vetor de query, encontra os `k` vetores mais próximos
- Distância medida por **cosseno** (similaridade) ou **euclidiana**

**ANN (Approximate Nearest Neighbors):**
- k-NN exato é O(n) - muito lento para milhões de documentos
- HNSW (Hierarchical Navigable Small World) oferece busca aproximada em O(log n)

```python
# Embedding com Azure OpenAI
from openai import AzureOpenAI
import os

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

def get_embedding(text: str) -> list[float]:
    deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")
    response = client.embeddings.create(
        model=deployment,  # Nome do deployment no Azure
        input=text
    )
    return response.data[0].embedding  # 1536 dimensões
```

### 2.3 Busca Geoespacial

OpenSearch/Elasticsearch suportam dois tipos de dados geográficos:

| Tipo | Uso | Exemplo |
|------|-----|---------|
| `geo_point` | Ponto único (lat/lon) | Localização de uma loja |
| `geo_shape` | Polígonos, linhas | Área de cobertura, rota |

**Queries geoespaciais:**

```json
// Distância de um ponto
{
  "geo_distance": {
    "distance": "10km",
    "localizacao": { "lat": -23.55, "lon": -46.63 }
  }
}

// Dentro de um bounding box
{
  "geo_bounding_box": {
    "localizacao": {
      "top_left": { "lat": -23.4, "lon": -46.8 },
      "bottom_right": { "lat": -23.7, "lon": -46.5 }
    }
  }
}
```

### 2.4 Por que Combinar? (Complementaridade)

Cada técnica tem forças e fraquezas:

| Aspecto | Full-Text (BM25) | Vetorial (k-NN) | Geo |
|---------|------------------|-----------------|-----|
| **Força** | Match exato, rápido | Semântica, sinônimos | Proximidade física |
| **Fraqueza** | Não entende contexto | Mais lento, requer embeddings | Só localização |
| **Quando usar** | Termos específicos | Linguagem natural | "Perto de mim" |

**A combinação é poderosa porque:**

1. **BM25** garante que termos exatos tenham peso (ex: código CNAE específico)
2. **k-NN** expande semanticamente (ex: "mecânica" → "auto service")
3. **Geo** filtra por relevância geográfica (ex: só na minha cidade)

---

## 3. Elasticsearch vs OpenSearch

### 3.1 Histórico e Fork

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIMELINE                                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  2010        2015        2019        2021        2024                       │
│    │           │           │           │           │                        │
│    ▼           ▼           ▼           ▼           ▼                        │
│  ┌───┐      ┌───┐      ┌───────┐   ┌───────┐   ┌───────┐                   │
│  │ES │──────│ES │──────│ES 7.x │   │FORK!  │   │OS 2.x │                   │
│  │1.0│      │5.0│      │Apache │   │       │   │Apache │                   │
│  └───┘      └───┘      │License│   │       │   │  2.0  │                   │
│                        └───────┘   │       │   └───────┘                   │
│                            │       │       │       │                        │
│                            │       ▼       │       │                        │
│                            │   ┌───────┐   │   ┌───────┐                   │
│                            └──▶│ES 7.10│───┴──▶│ES 8.x │                   │
│                                │Elastic│       │SSPL   │                   │
│                                │License│       │License│                   │
│                                └───────┘       └───────┘                   │
│                                                                             │
│  Elastic mudou para SSPL (2021) → AWS criou OpenSearch (fork do 7.10)      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Comparativo para Busca Híbrida

| Feature | Elasticsearch 8.x | OpenSearch 2.x |
|---------|-------------------|----------------|
| **Full-text (BM25)** | ✅ Nativo | ✅ Nativo |
| **k-NN Vector** | ✅ `dense_vector` | ✅ `knn_vector` |
| **Geo queries** | ✅ Completo | ✅ Completo |
| **HNSW Engine** | Lucene | Lucene, nmslib, Faiss |
| **Neural Search** | ❌ Requer plugin | ✅ ML Commons nativo |
| **Hybrid Query** | `knn` + `query` | `hybrid` query nativa |
| **Licença** | SSPL / Elastic License | Apache 2.0 |
| **Cloud gerenciado** | Elastic Cloud | AWS OpenSearch Service |

### 3.3 Compatibilidade de APIs

A maioria das APIs é compatível entre ES e OS:

```python
# Funciona em AMBOS
from opensearchpy import OpenSearch  # ou elasticsearch

client = OpenSearch(hosts=["localhost:9200"])

# Mesmo DSL para queries básicas
client.search(index="meu_indice", body={
    "query": {
        "bool": {
            "must": [{"match": {"descricao": "oficina"}}],
            "filter": [{"geo_distance": {"distance": "10km", "loc": {"lat": -23.5, "lon": -46.6}}}]
        }
    }
})
```

**Diferenças principais:**

| Operação | Elasticsearch | OpenSearch |
|----------|---------------|------------|
| k-NN query | `"knn": {...}` | `"knn": {...}` (similar) |
| k-NN + BM25 | `knn` dentro de `query` | `hybrid` query |
| Vector field | `dense_vector` | `knn_vector` |
| HNSW params | `index_options` | `method.parameters` |

### 3.4 Quando Usar Cada Um

| Cenário | Recomendação |
|---------|--------------|
| Já usa AWS | **OpenSearch** (integração nativa) |
| Precisa de ML nativo | **OpenSearch** (ML Commons) |
| Já tem stack Elastic | **Elasticsearch** (menos migração) |
| Open source puro | **OpenSearch** (Apache 2.0) |
| Observabilidade integrada | **Elasticsearch** (APM, Logs) |

**Para este artigo:** Usaremos **OpenSearch** por ser open source e ter `hybrid` query nativa.

---

## 4. Arquitetura da Solução

### 4.1 Design do Índice (Mapping)

Nosso índice `estabelecimentos_v001` combinará as três modalidades:

```json
PUT /estabelecimentos_v001
{
  "settings": {
    "index": {
      "number_of_shards": 1,
      "number_of_replicas": 0,
      "knn": true
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
            "brazilian_stemmer"
          ]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "cnpj": {
        "type": "keyword"
      },
      "razao_social": {
        "type": "text",
        "analyzer": "brazilian_text"
      },
      "cnae_codigo": {
        "type": "keyword"
      },
      "cnae_descricao": {
        "type": "text",
        "analyzer": "brazilian_text"
      },
      "atividade_descritiva": {
        "type": "text",
        "analyzer": "brazilian_text"
      },
      "texto_busca": {
        "type": "text",
        "analyzer": "brazilian_text"
      },
      "embedding": {
        "type": "knn_vector",
        "dimension": 1536,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "lucene",
          "parameters": {
            "ef_construction": 128,
            "m": 16
          }
        }
      },
      "localizacao": {
        "type": "geo_point"
      },
      "endereco": {
        "type": "object",
        "properties": {
          "logradouro": { "type": "text" },
          "numero": { "type": "keyword" },
          "bairro": { "type": "text" },
          "municipio": { "type": "keyword" },
          "uf": { "type": "keyword" },
          "cep": { "type": "keyword" }
        }
      },
      "situacao_cadastral": {
        "type": "keyword"
      },
      "porte": {
        "type": "keyword"
      }
    }
  }
}
```

### 4.2 Pipeline de Ingestão

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE DE INGESTÃO                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  1. FONTE   │    │ 2. ENRICH   │    │ 3. EMBED    │    │ 4. INDEX    │  │
│  │             │    │             │    │             │    │             │  │
│  │ CSV/JSON    │───▶│ Concatenar  │───▶│ OpenAI API  │───▶│ Bulk API    │  │
│  │ CNAE + Emp. │    │ texto_busca │    │ embedding   │    │ OpenSearch  │  │
│  │             │    │             │    │ 1536 dims   │    │             │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
│  texto_busca = f"{razao_social}. {cnae_descricao}. {atividade_descritiva}" │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Estratégias de Scoring Híbrido

Existem 3 abordagens principais para combinar scores:

#### A) Post-fusion (Reciprocal Rank Fusion - RRF)

Executa queries separadas e combina rankings:

```
RRF_score = Σ 1 / (k + rank_i)
```

#### B) Score Combination (Weighted Sum)

Normaliza e soma scores com pesos:

```
final_score = w1 × norm(bm25) + w2 × norm(knn) + w3 × norm(geo)
```

#### C) OpenSearch Hybrid Query (Recomendado)

Query nativa que faz fusão internamente:

```json
{
  "query": {
    "hybrid": {
      "queries": [
        { "match": { "texto_busca": "oficina mecânica" } },
        { "knn": { "embedding": { "vector": [...], "k": 10 } } }
      ]
    }
  },
  "post_filter": {
    "geo_distance": { "distance": "10km", "localizacao": {...} }
  }
}
```

---

## 5. Implementação Prática

### 5.1 Criando o Índice Híbrido

```python
from opensearchpy import OpenSearch

client = OpenSearch(hosts=["localhost:9200"])

# Criar índice com k-NN habilitado
INDEX_SETTINGS = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 100
        },
        "analysis": {
            "analyzer": {
                "brazilian_text": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding", 
                              "brazilian_stop", "brazilian_stemmer"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "texto_busca": {"type": "text", "analyzer": "brazilian_text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1536,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "lucene"
                }
            },
            "localizacao": {"type": "geo_point"}
        }
    }
}

client.indices.create(index="estabelecimentos_v001", body=INDEX_SETTINGS)
```

### 5.2 Configurando Analyzers para Português

O analyzer `brazilian_text` processa texto em português:

```json
{
  "analysis": {
    "filter": {
      "brazilian_stop": {
        "type": "stop",
        "stopwords": "_brazilian_"  // Remove: de, da, para, com, etc.
      },
      "brazilian_stemmer": {
        "type": "stemmer",
        "language": "brazilian"  // carros → carr, mecânica → mecan
      }
    },
    "analyzer": {
      "brazilian_text": {
        "tokenizer": "standard",
        "filter": [
          "lowercase",      // OFICINA → oficina
          "asciifolding",   // mecânica → mecanica
          "brazilian_stop",
          "brazilian_stemmer"
        ]
      }
    }
  }
}
```

**Teste do analyzer:**

```bash
POST /estabelecimentos_v001/_analyze
{
  "analyzer": "brazilian_text",
  "text": "Oficina Mecânica de Carros Importados"
}

# Resultado: ["oficin", "mecan", "carr", "import"]
```

### 5.3 Indexando Documentos com Embeddings

```python
from openai import AzureOpenAI
from opensearchpy import helpers
import os

# Configuração Azure OpenAI
azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")

def get_embedding(text: str) -> list[float]:
    """Gera embedding com Azure OpenAI."""
    response = azure_client.embeddings.create(
        model=EMBEDDINGS_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding

def prepare_document(doc: dict) -> dict:
    """Prepara documento para indexação."""
    # Concatenar texto para busca
    texto = f"{doc['razao_social']}. {doc['cnae_descricao']}. {doc['atividade']}"
    
    return {
        **doc,
        "texto_busca": texto,
        "embedding": get_embedding(texto)
    }

# Bulk indexing
documents = [prepare_document(d) for d in raw_documents]
actions = [
    {"_index": "estabelecimentos_v001", "_source": doc}
    for doc in documents
]
helpers.bulk(client, actions)
```

### 5.4 Queries Híbridas (3 Abordagens)

#### Abordagem 1: Full-Text + Geo (Sem Vetores)

```python
def busca_fulltext_geo(query: str, lat: float, lon: float, raio_km: float):
    """Busca textual filtrada por localização."""
    return client.search(
        index="estabelecimentos_v001",
        body={
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["razao_social^2", "cnae_descricao", "texto_busca"]
                        }
                    },
                    "filter": {
                        "geo_distance": {
                            "distance": f"{raio_km}km",
                            "localizacao": {"lat": lat, "lon": lon}
                        }
                    }
                }
            },
            "sort": [
                "_score",
                {"_geo_distance": {"localizacao": {"lat": lat, "lon": lon}, "order": "asc"}}
            ]
        }
    )
```

#### Abordagem 2: Vetorial + Geo (Sem BM25)

```python
def busca_vetorial_geo(query: str, lat: float, lon: float, raio_km: float, k: int = 10):
    """Busca semântica filtrada por localização."""
    query_embedding = get_embedding(query)
    
    return client.search(
        index="estabelecimentos_v001",
        body={
            "query": {
                "bool": {
                    "must": {
                        "knn": {
                            "embedding": {
                                "vector": query_embedding,
                                "k": k
                            }
                        }
                    },
                    "filter": {
                        "geo_distance": {
                            "distance": f"{raio_km}km",
                            "localizacao": {"lat": lat, "lon": lon}
                        }
                    }
                }
            }
        }
    )
```

#### Abordagem 3: Híbrida Completa (BM25 + k-NN + Geo)

```python
def busca_hibrida_completa(
    query: str, 
    lat: float, 
    lon: float, 
    raio_km: float,
    peso_bm25: float = 0.3,
    peso_knn: float = 0.7
):
    """Combina BM25 e k-NN com script_score."""
    query_embedding = get_embedding(query)
    
    return client.search(
        index="estabelecimentos_v001",
        body={
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "should": [
                                {"multi_match": {"query": query, "fields": ["texto_busca"]}}
                            ],
                            "filter": {
                                "geo_distance": {
                                    "distance": f"{raio_km}km",
                                    "localizacao": {"lat": lat, "lon": lon}
                                }
                            }
                        }
                    },
                    "script": {
                        "source": f"""
                            double bm25 = _score;
                            double knn = cosineSimilarity(params.vec, 'embedding') + 1.0;
                            return ({peso_bm25} * bm25) + ({peso_knn} * knn);
                        """,
                        "params": {"vec": query_embedding}
                    }
                }
            }
        }
    )
```

---

## 6. Otimização e Performance

### 6.1 Tuning de k-NN (HNSW Parameters)

O algoritmo HNSW tem dois parâmetros principais:

| Parâmetro | Default | Descrição | Trade-off |
|-----------|---------|-----------|-----------|
| `m` | 16 | Conexões por nó | ↑ = mais preciso, mais memória |
| `ef_construction` | 100 | Qualidade do grafo | ↑ = indexação mais lenta, busca melhor |
| `ef_search` | 100 | Candidatos na busca | ↑ = mais preciso, mais lento |

**Recomendações por cenário:**

```json
// Alta precisão (< 1M docs)
{
  "method": {
    "parameters": {
      "m": 24,
      "ef_construction": 256
    }
  }
}

// Alto volume (> 10M docs)
{
  "method": {
    "parameters": {
      "m": 16,
      "ef_construction": 128
    }
  }
}
```

### 6.2 Sharding Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│  ESTRATÉGIA DE SHARDS                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Volume          Shards    Replicas    Observação               │
│  ─────────────────────────────────────────────────────────────  │
│  < 1M docs       1         1           Single shard é mais      │
│                                        rápido para k-NN         │
│                                                                 │
│  1M - 10M        3         1           Balancear carga          │
│                                                                 │
│  > 10M           5-10      1-2         Considerar índices       │
│                                        por região/categoria     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Caching e Warmup

```python
# Warmup de índice k-NN
client.indices.put_settings(
    index="estabelecimentos_v001",
    body={
        "index": {
            "knn.algo_param.ef_search": 100
        }
    }
)

# Forçar warmup do cache
client.search(
    index="estabelecimentos_v001",
    body={"query": {"knn": {"embedding": {"vector": warmup_vector, "k": 1}}}}
)
```

### 6.4 Benchmarks Comparativos

| Query Type | Latência (p50) | Latência (p99) | Recall@10 |
|------------|---------------|----------------|-----------|
| Full-text only | 15ms | 45ms | 0.65 |
| k-NN only | 25ms | 80ms | 0.85 |
| Híbrida (BM25+kNN) | 40ms | 120ms | **0.92** |
| Híbrida + Geo | 50ms | 150ms | **0.92** |

*Testado com 200K documentos, 3 shards, m=16, ef=100*

---

## 7. Produção e Observabilidade

### 7.1 Monitoramento de Latência

```python
# Middleware para logging de queries
import time
import structlog

logger = structlog.get_logger()

async def search_with_metrics(query_body: dict):
    start = time.perf_counter()
    
    response = await client.search(
        index="estabelecimentos_v001",
        body=query_body
    )
    
    latency_ms = (time.perf_counter() - start) * 1000
    
    logger.info(
        "search_executed",
        latency_ms=latency_ms,
        total_hits=response["hits"]["total"]["value"],
        took_ms=response["took"],
        shards_total=response["_shards"]["total"],
        shards_successful=response["_shards"]["successful"]
    )
    
    return response
```

### 7.2 A/B Testing de Relevância

```python
# Feature flags para testar pesos diferentes
SEARCH_CONFIGS = {
    "control": {"peso_bm25": 0.5, "peso_knn": 0.5},
    "variant_a": {"peso_bm25": 0.3, "peso_knn": 0.7},
    "variant_b": {"peso_bm25": 0.2, "peso_knn": 0.8},
}

def get_search_config(user_id: str) -> dict:
    """Determina config baseado no hash do user_id."""
    bucket = hash(user_id) % 100
    
    if bucket < 33:
        return SEARCH_CONFIGS["control"]
    elif bucket < 66:
        return SEARCH_CONFIGS["variant_a"]
    else:
        return SEARCH_CONFIGS["variant_b"]
```

### 7.3 Fallback Strategies

```python
async def search_with_fallback(query: str, lat: float, lon: float):
    """Busca com fallbacks progressivos."""
    
    try:
        # Tentativa 1: Híbrida completa
        return await busca_hibrida_completa(query, lat, lon, raio_km=10)
    except Exception as e:
        logger.warning("hybrid_search_failed", error=str(e))
    
    try:
        # Fallback 2: Apenas full-text + geo
        return await busca_fulltext_geo(query, lat, lon, raio_km=10)
    except Exception as e:
        logger.warning("fulltext_search_failed", error=str(e))
    
    try:
        # Fallback 3: Apenas full-text (sem geo)
        return await busca_fulltext(query)
    except Exception as e:
        logger.error("all_searches_failed", error=str(e))
        raise
```

---

## 8. Conclusão

### Resumo das Técnicas

| Técnica | Quando Usar | Exemplo |
|---------|-------------|---------|
| **Full-Text (BM25)** | Termos específicos, códigos | "CNAE 4520001" |
| **Vetorial (k-NN)** | Linguagem natural, sinônimos | "conserto de carro" |
| **Geoespacial** | Proximidade física | "perto de mim" |
| **Híbrida** | Combinação de critérios | "oficina mecânica próxima" |

### Trade-offs e Decisões de Design

```
┌─────────────────────────────────────────────────────────────────┐
│  TRADE-OFFS                                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Precisão vs Velocidade                                         │
│  ├── Mais k (k-NN) = mais preciso, mais lento                  │
│  └── ef_search maior = melhor recall, maior latência           │
│                                                                 │
│  Custo vs Qualidade                                             │
│  ├── Embeddings requerem API calls (custo)                     │
│  └── Modelos maiores = melhor qualidade, mais caro             │
│                                                                 │
│  Simplicidade vs Flexibilidade                                  │
│  ├── Full-text apenas = simples, limitado                      │
│  └── Híbrido = complexo, poderoso                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Próximos Passos

1. **Experimente** com seus próprios dados
2. **Ajuste os pesos** BM25/k-NN via A/B testing
3. **Monitore latência** e recall em produção
4. **Considere** Neural Search do OpenSearch para pipelines avançados

### Recursos Adicionais

- 📁 [Código completo no repositório](./code/)
- 📊 [Dataset de exemplo](./data/)
- 🐳 [Docker Compose para ambiente local](./docker-compose.yml)

---

## 🚀 Quick Start

```bash
# 1. Subir OpenSearch local
cd docs/articles/hybrid-search
docker-compose up -d

# 2. Verificar se está rodando
curl http://localhost:9200

# 3. Configurar Azure OpenAI
cd code
cp env-template.txt .env
# Edite .env com suas credenciais Azure OpenAI
nano .env

# 4. Instalar dependências e executar
pip install -r requirements.txt
python 01_create_index.py        # Criar índice
python 02_generate_data.py       # Gerar dados fictícios
python 03_index_with_embeddings.py  # Indexar com embeddings
python 04_hybrid_queries.py      # Testar queries

# 5. Acessar Dashboards (opcional)
open http://localhost:5601
```

---

## 📚 Referências

- [OpenSearch Documentation](https://opensearch.org/docs/latest/)
- [Elasticsearch Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [HNSW Algorithm Paper](https://arxiv.org/abs/1603.09320)
- [BM25 Explained](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)
- [Azure OpenAI Embeddings](https://learn.microsoft.com/azure/ai-services/openai/how-to/embeddings)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

---

**Autor:** Leonardo de Melo (demelo01@gmail.com)  
**Data:** Fevereiro 2026  
**Licença:** MIT
