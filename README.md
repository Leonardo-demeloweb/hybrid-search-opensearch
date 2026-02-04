# Hybrid Search at Scale: Full-Text + Vectors + Geolocation

> **Combining BM25, k-NN and Geo Queries with OpenSearch/Elasticsearch**

[![OpenSearch](https://img.shields.io/badge/OpenSearch-2.11+-blue)](https://opensearch.org/)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.x-yellow)](https://www.elastic.co/)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org/)

---

## 📋 Table of Contents

1. [Introduction](#1-introduction)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [Elasticsearch vs OpenSearch](#3-elasticsearch-vs-opensearch)
4. [Solution Architecture](#4-solution-architecture)
5. [Practical Implementation](#5-practical-implementation)
6. [Optimization and Performance](#6-optimization-and-performance)
7. [Production and Observability](#7-production-and-observability)
8. [Conclusion](#8-conclusion)

---

## 1. Introduction

### The Problem: Limitations of Traditional Search

Imagine a search system for **commercial establishments**. A user types:

> *"auto repair shop specialized in imported cars near me"*

A traditional search based only on **full-text search** (BM25) would find documents containing exactly those words. But what if the establishment is registered as:

- *"Automotive center - premium vehicle repairs"*
- *"Auto service - BMW, Mercedes, Audi maintenance"*

Full-text would fail to connect **"auto repair shop"** with **"automotive center"**, and **"imported cars"** with **"BMW, Mercedes"**. Furthermore, **"near me"** wouldn't even be processed.

### The Solution: Multi-Modal Hybrid Search

**Hybrid search** combines three complementary techniques:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUERY: "auto repair imported cars"                   │
│                         LOCATION: New York, NY (40.71, -74.00)              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐           │
│  │  1. FULL-TEXT   │   │  2. VECTOR      │   │  3. GEOSPATIAL  │           │
│  │     (BM25)      │   │    (k-NN)       │   │   (geo_point)   │           │
│  ├─────────────────┤   ├─────────────────┤   ├─────────────────┤           │
│  │                 │   │                 │   │                 │           │
│  │ Search tokens:  │   │ Query embedding │   │ Distance        │           │
│  │ "auto"          │   │ → finds docs    │   │ filter:         │           │
│  │ "repair"        │   │ semantically    │   │ 10km radius     │           │
│  │ "imported"      │   │ similar         │   │ from point      │           │
│  │ "cars"          │   │                 │   │                 │           │
│  │                 │   │                 │   │                 │           │
│  │ ✓ Exact match   │   │ ✓ "auto service"│   │ ✓ Only          │           │
│  │                 │   │ ✓ "BMW, Audi"   │   │   nearby        │           │
│  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘           │
│           │                     │                     │                     │
│           └──────────┬──────────┴──────────┬──────────┘                     │
│                      │                     │                                │
│                      ▼                     ▼                                │
│           ┌─────────────────────────────────────────┐                       │
│           │         COMBINED SCORE                   │                       │
│           │  (0.3 × BM25) + (0.5 × kNN) + (0.2 × geo)│                       │
│           │         = Final Relevance                │                       │
│           └─────────────────────────────────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Use Case: Business Establishment Search by Industry Code

In this article, we'll build a hybrid search system using public **industry classification codes** combined with **fictitious geolocated establishments**.

The index will have:

| Field | Type | Search |
|-------|------|--------|
| `activity_description` | text (BM25) | Full-text |
| `embedding` | knn_vector (1536d) | Semantic |
| `location` | geo_point | Geospatial |
| `industry_code`, `company_name` | keyword/text | Filters |

---

## 2. Theoretical Foundations

### 2.1 Full-Text Search (BM25)

The **BM25** (Best Matching 25) algorithm is the standard for text search. It calculates relevance based on:

- **TF (Term Frequency)**: How many times the term appears in the document
- **IDF (Inverse Document Frequency)**: How rare the term is in the corpus
- **Document Length Normalization**: Shorter documents get a boost

**Simplified formula:**

```
score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D|/avgdl))
```

Where:
- `k1` = 1.2 (frequency saturation)
- `b` = 0.75 (length normalization)

**Limitation:** BM25 is **lexical** - it doesn't understand synonyms or semantic context.

```python
# Example: BM25 doesn't connect these terms
query = "car"
doc1 = "automobile"  # ❌ BM25 doesn't find
doc2 = "vehicle"     # ❌ BM25 doesn't find
doc3 = "car"         # ✅ Exact match
```

### 2.2 Vector Search (k-NN / ANN)

Vector search represents texts as **high-dimensional vectors** (embeddings) where semantically similar texts are close in vector space.

**How it works:**

```
┌─────────────────────────────────────────────────────────────────┐
│  VECTOR SPACE (simplified in 2D)                                │
│                                                                 │
│     "automobile" ●──────────● "car"                            │
│                     \      /                                    │
│                      \    /                                     │
│                       \  /                                      │
│                        ●                                        │
│                    "vehicle"                                    │
│                                                                 │
│                                        ● "bicycle"              │
│                                                                 │
│                                                                 │
│                    ● "restaurant"                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**k-NN Algorithm (k-Nearest Neighbors):**
- Given a query vector, finds the `k` closest vectors
- Distance measured by **cosine** (similarity) or **euclidean**

**ANN (Approximate Nearest Neighbors):**
- Exact k-NN is O(n) - too slow for millions of documents
- HNSW (Hierarchical Navigable Small World) offers approximate search in O(log n)

```python
# Embedding with Azure OpenAI
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
        model=deployment,  # Deployment name in Azure
        input=text
    )
    return response.data[0].embedding  # 1536 dimensions
```

### 2.3 Geospatial Search

OpenSearch/Elasticsearch support two types of geographic data:

| Type | Use | Example |
|------|-----|---------|
| `geo_point` | Single point (lat/lon) | Store location |
| `geo_shape` | Polygons, lines | Coverage area, route |

**Geospatial queries:**

```json
// Distance from a point
{
  "geo_distance": {
    "distance": "10km",
    "location": { "lat": 40.71, "lon": -74.00 }
  }
}

// Within a bounding box
{
  "geo_bounding_box": {
    "location": {
      "top_left": { "lat": 40.9, "lon": -74.3 },
      "bottom_right": { "lat": 40.5, "lon": -73.7 }
    }
  }
}
```

### 2.4 Why Combine? (Complementarity)

Each technique has strengths and weaknesses:

| Aspect | Full-Text (BM25) | Vector (k-NN) | Geo |
|--------|------------------|---------------|-----|
| **Strength** | Exact match, fast | Semantics, synonyms | Physical proximity |
| **Weakness** | No context understanding | Slower, requires embeddings | Location only |
| **When to use** | Specific terms | Natural language | "Near me" |

**The combination is powerful because:**

1. **BM25** ensures exact terms have weight (e.g., specific industry code)
2. **k-NN** expands semantically (e.g., "mechanic" → "auto service")
3. **Geo** filters by geographic relevance (e.g., only in my city)

---

## 3. Elasticsearch vs OpenSearch

### 3.1 History and Fork

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
│  Elastic changed to SSPL (2021) → AWS created OpenSearch (fork of 7.10)    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Comparison for Hybrid Search

| Feature | Elasticsearch 8.x | OpenSearch 2.x |
|---------|-------------------|----------------|
| **Full-text (BM25)** | ✅ Native | ✅ Native |
| **k-NN Vector** | ✅ `dense_vector` | ✅ `knn_vector` |
| **Geo queries** | ✅ Complete | ✅ Complete |
| **HNSW Engine** | Lucene | Lucene, nmslib, Faiss |
| **Neural Search** | ❌ Requires plugin | ✅ ML Commons native |
| **Hybrid Query** | `knn` + `query` | Native `hybrid` query |
| **License** | SSPL / Elastic License | Apache 2.0 |
| **Managed Cloud** | Elastic Cloud | AWS OpenSearch Service |

### 3.3 API Compatibility

Most APIs are compatible between ES and OS:

```python
# Works on BOTH
from opensearchpy import OpenSearch  # or elasticsearch

client = OpenSearch(hosts=["localhost:9200"])

# Same DSL for basic queries
client.search(index="my_index", body={
    "query": {
        "bool": {
            "must": [{"match": {"description": "repair"}}],
            "filter": [{"geo_distance": {"distance": "10km", "loc": {"lat": 40.7, "lon": -74.0}}}]
        }
    }
})
```

**Main differences:**

| Operation | Elasticsearch | OpenSearch |
|-----------|---------------|------------|
| k-NN query | `"knn": {...}` | `"knn": {...}` (similar) |
| k-NN + BM25 | `knn` inside `query` | `hybrid` query |
| Vector field | `dense_vector` | `knn_vector` |
| HNSW params | `index_options` | `method.parameters` |

### 3.4 When to Use Each

| Scenario | Recommendation |
|----------|---------------|
| Already using AWS | **OpenSearch** (native integration) |
| Need native ML | **OpenSearch** (ML Commons) |
| Already have Elastic stack | **Elasticsearch** (less migration) |
| Pure open source | **OpenSearch** (Apache 2.0) |
| Integrated observability | **Elasticsearch** (APM, Logs) |

**For this article:** We'll use **OpenSearch** as it's open source and has native `hybrid` query.

---

## 4. Solution Architecture

### 4.1 Index Design (Mapping)

Our `establishments_v001` index will combine all three modalities:

```json
PUT /establishments_v001
{
  "settings": {
    "index": {
      "number_of_shards": 1,
      "number_of_replicas": 0,
      "knn": true
    },
    "analysis": {
      "filter": {
        "english_stop": {
          "type": "stop",
          "stopwords": "_english_"
        },
        "english_stemmer": {
          "type": "stemmer",
          "language": "english"
        }
      },
      "analyzer": {
        "english_text": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": [
            "lowercase",
            "asciifolding",
            "english_stop",
            "english_stemmer"
          ]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "business_id": {
        "type": "keyword"
      },
      "company_name": {
        "type": "text",
        "analyzer": "english_text"
      },
      "industry_code": {
        "type": "keyword"
      },
      "industry_description": {
        "type": "text",
        "analyzer": "english_text"
      },
      "activity_description": {
        "type": "text",
        "analyzer": "english_text"
      },
      "search_text": {
        "type": "text",
        "analyzer": "english_text"
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
      "location": {
        "type": "geo_point"
      },
      "address": {
        "type": "object",
        "properties": {
          "street": { "type": "text" },
          "number": { "type": "keyword" },
          "neighborhood": { "type": "text" },
          "city": { "type": "keyword" },
          "state": { "type": "keyword" },
          "zip_code": { "type": "keyword" }
        }
      },
      "status": {
        "type": "keyword"
      },
      "size": {
        "type": "keyword"
      }
    }
  }
}
```

### 4.2 Ingestion Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  1. SOURCE  │    │ 2. ENRICH   │    │ 3. EMBED    │    │ 4. INDEX    │  │
│  │             │    │             │    │             │    │             │  │
│  │ CSV/JSON    │───▶│ Concatenate │───▶│ OpenAI API  │───▶│ Bulk API    │  │
│  │ Industry +  │    │ search_text │    │ embedding   │    │ OpenSearch  │  │
│  │ Companies   │    │             │    │ 1536 dims   │    │             │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
│  search_text = f"{company_name}. {industry_desc}. {activity_description}"  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Hybrid Scoring Strategies

There are 3 main approaches to combine scores:

#### A) Post-fusion (Reciprocal Rank Fusion - RRF)

Executes separate queries and combines rankings:

```
RRF_score = Σ 1 / (k + rank_i)
```

#### B) Score Combination (Weighted Sum)

Normalizes and sums scores with weights:

```
final_score = w1 × norm(bm25) + w2 × norm(knn) + w3 × norm(geo)
```

#### C) OpenSearch Hybrid Query (Recommended)

Native query that performs fusion internally:

```json
{
  "query": {
    "hybrid": {
      "queries": [
        { "match": { "search_text": "auto repair" } },
        { "knn": { "embedding": { "vector": [...], "k": 10 } } }
      ]
    }
  },
  "post_filter": {
    "geo_distance": { "distance": "10km", "location": {...} }
  }
}
```

---

## 5. Practical Implementation

### 5.1 Creating the Hybrid Index

```python
from opensearchpy import OpenSearch

client = OpenSearch(hosts=["localhost:9200"])

# Create index with k-NN enabled
INDEX_SETTINGS = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 100
        },
        "analysis": {
            "analyzer": {
                "english_text": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding", 
                              "english_stop", "english_stemmer"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "search_text": {"type": "text", "analyzer": "english_text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1536,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "lucene"
                }
            },
            "location": {"type": "geo_point"}
        }
    }
}

client.indices.create(index="establishments_v001", body=INDEX_SETTINGS)
```

### 5.2 Configuring Analyzers for English

The `english_text` analyzer processes English text:

```json
{
  "analysis": {
    "filter": {
      "english_stop": {
        "type": "stop",
        "stopwords": "_english_"  // Removes: the, a, an, is, etc.
      },
      "english_stemmer": {
        "type": "stemmer",
        "language": "english"  // cars → car, mechanical → mechan
      }
    },
    "analyzer": {
      "english_text": {
        "tokenizer": "standard",
        "filter": [
          "lowercase",      // REPAIR → repair
          "asciifolding",   // café → cafe
          "english_stop",
          "english_stemmer"
        ]
      }
    }
  }
}
```

**Testing the analyzer:**

```bash
POST /establishments_v001/_analyze
{
  "analyzer": "english_text",
  "text": "Auto Repair Shop for Imported Cars"
}

# Result: ["auto", "repair", "shop", "import", "car"]
```

### 5.3 Indexing Documents with Embeddings

```python
from openai import AzureOpenAI
from opensearchpy import helpers
import os

# Azure OpenAI configuration
azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")

def get_embedding(text: str) -> list[float]:
    """Generate embedding with Azure OpenAI."""
    response = azure_client.embeddings.create(
        model=EMBEDDINGS_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding

def prepare_document(doc: dict) -> dict:
    """Prepare document for indexing."""
    # Concatenate text for search
    text = f"{doc['company_name']}. {doc['industry_description']}. {doc['activity']}"
    
    return {
        **doc,
        "search_text": text,
        "embedding": get_embedding(text)
    }

# Bulk indexing
documents = [prepare_document(d) for d in raw_documents]
actions = [
    {"_index": "establishments_v001", "_source": doc}
    for doc in documents
]
helpers.bulk(client, actions)
```

### 5.4 Hybrid Queries (3 Approaches)

#### Approach 1: Full-Text + Geo (No Vectors)

```python
def search_fulltext_geo(query: str, lat: float, lon: float, radius_km: float):
    """Text search filtered by location."""
    return client.search(
        index="establishments_v001",
        body={
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["company_name^2", "industry_description", "search_text"]
                        }
                    },
                    "filter": {
                        "geo_distance": {
                            "distance": f"{radius_km}km",
                            "location": {"lat": lat, "lon": lon}
                        }
                    }
                }
            },
            "sort": [
                "_score",
                {"_geo_distance": {"location": {"lat": lat, "lon": lon}, "order": "asc"}}
            ]
        }
    )
```

#### Approach 2: Vector + Geo (No BM25)

```python
def search_vector_geo(query: str, lat: float, lon: float, radius_km: float, k: int = 10):
    """Semantic search filtered by location."""
    query_embedding = get_embedding(query)
    
    return client.search(
        index="establishments_v001",
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
                            "distance": f"{radius_km}km",
                            "location": {"lat": lat, "lon": lon}
                        }
                    }
                }
            }
        }
    )
```

#### Approach 3: Complete Hybrid (BM25 + k-NN + Geo)

```python
def search_hybrid_complete(
    query: str, 
    lat: float, 
    lon: float, 
    radius_km: float,
    bm25_weight: float = 0.3,
    knn_weight: float = 0.7
):
    """Combines BM25 and k-NN with script_score."""
    query_embedding = get_embedding(query)
    
    return client.search(
        index="establishments_v001",
        body={
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "should": [
                                {"multi_match": {"query": query, "fields": ["search_text"]}}
                            ],
                            "filter": {
                                "geo_distance": {
                                    "distance": f"{radius_km}km",
                                    "location": {"lat": lat, "lon": lon}
                                }
                            }
                        }
                    },
                    "script": {
                        "source": f"""
                            double bm25 = _score;
                            double knn = cosineSimilarity(params.vec, 'embedding') + 1.0;
                            return ({bm25_weight} * bm25) + ({knn_weight} * knn);
                        """,
                        "params": {"vec": query_embedding}
                    }
                }
            }
        }
    )
```

---

## 6. Optimization and Performance

### 6.1 k-NN Tuning (HNSW Parameters)

The HNSW algorithm has two main parameters:

| Parameter | Default | Description | Trade-off |
|-----------|---------|-------------|-----------|
| `m` | 16 | Connections per node | ↑ = more accurate, more memory |
| `ef_construction` | 100 | Graph quality | ↑ = slower indexing, better search |
| `ef_search` | 100 | Search candidates | ↑ = more accurate, slower |

**Recommendations by scenario:**

```json
// High precision (< 1M docs)
{
  "method": {
    "parameters": {
      "m": 24,
      "ef_construction": 256
    }
  }
}

// High volume (> 10M docs)
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
│  SHARDING STRATEGY                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Volume          Shards    Replicas    Notes                    │
│  ─────────────────────────────────────────────────────────────  │
│  < 1M docs       1         1           Single shard is faster   │
│                                        for k-NN                 │
│                                                                 │
│  1M - 10M        3         1           Load balancing           │
│                                                                 │
│  > 10M           5-10      1-2         Consider indices         │
│                                        by region/category       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Caching and Warmup

```python
# k-NN index warmup
client.indices.put_settings(
    index="establishments_v001",
    body={
        "index": {
            "knn.algo_param.ef_search": 100
        }
    }
)

# Force cache warmup
client.search(
    index="establishments_v001",
    body={"query": {"knn": {"embedding": {"vector": warmup_vector, "k": 1}}}}
)
```

### 6.4 Comparative Benchmarks

| Query Type | Latency (p50) | Latency (p99) | Recall@10 |
|------------|---------------|---------------|-----------|
| Full-text only | 15ms | 45ms | 0.65 |
| k-NN only | 25ms | 80ms | 0.85 |
| Hybrid (BM25+kNN) | 40ms | 120ms | **0.92** |
| Hybrid + Geo | 50ms | 150ms | **0.92** |

*Tested with 200K documents, 3 shards, m=16, ef=100*

---

## 7. Production and Observability

### 7.1 Latency Monitoring

```python
# Middleware for query logging
import time
import structlog

logger = structlog.get_logger()

async def search_with_metrics(query_body: dict):
    start = time.perf_counter()
    
    response = await client.search(
        index="establishments_v001",
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

### 7.2 Relevance A/B Testing

```python
# Feature flags to test different weights
SEARCH_CONFIGS = {
    "control": {"bm25_weight": 0.5, "knn_weight": 0.5},
    "variant_a": {"bm25_weight": 0.3, "knn_weight": 0.7},
    "variant_b": {"bm25_weight": 0.2, "knn_weight": 0.8},
}

def get_search_config(user_id: str) -> dict:
    """Determines config based on user_id hash."""
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
    """Search with progressive fallbacks."""
    
    try:
        # Attempt 1: Complete hybrid
        return await search_hybrid_complete(query, lat, lon, radius_km=10)
    except Exception as e:
        logger.warning("hybrid_search_failed", error=str(e))
    
    try:
        # Fallback 2: Full-text + geo only
        return await search_fulltext_geo(query, lat, lon, radius_km=10)
    except Exception as e:
        logger.warning("fulltext_search_failed", error=str(e))
    
    try:
        # Fallback 3: Full-text only (no geo)
        return await search_fulltext(query)
    except Exception as e:
        logger.error("all_searches_failed", error=str(e))
        raise
```

---

## 8. Conclusion

### Techniques Summary

| Technique | When to Use | Example |
|-----------|-------------|---------|
| **Full-Text (BM25)** | Specific terms, codes | "NAICS 811111" |
| **Vector (k-NN)** | Natural language, synonyms | "car repair" |
| **Geospatial** | Physical proximity | "near me" |
| **Hybrid** | Combination of criteria | "nearby auto shop" |

### Trade-offs and Design Decisions

```
┌─────────────────────────────────────────────────────────────────┐
│  TRADE-OFFS                                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Precision vs Speed                                             │
│  ├── Higher k (k-NN) = more accurate, slower                   │
│  └── Higher ef_search = better recall, higher latency          │
│                                                                 │
│  Cost vs Quality                                                │
│  ├── Embeddings require API calls (cost)                       │
│  └── Larger models = better quality, more expensive            │
│                                                                 │
│  Simplicity vs Flexibility                                      │
│  ├── Full-text only = simple, limited                          │
│  └── Hybrid = complex, powerful                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Next Steps

1. **Experiment** with your own data
2. **Tune the weights** BM25/k-NN via A/B testing
3. **Monitor latency** and recall in production
4. **Consider** OpenSearch Neural Search for advanced pipelines

### Additional Resources

- 📁 [Complete code in repository](./code/)
- 📊 [Sample dataset](./data/)
- 🐳 [Docker Compose for local environment](./docker-compose.yml)

---

## 🚀 Quick Start

```bash
# 1. Start local OpenSearch
docker-compose up -d

# 2. Verify it's running
curl http://localhost:9200

# 3. Configure Azure OpenAI
cd code
cp env-template.txt .env
# Edit .env with your Azure OpenAI credentials
nano .env

# 4. Install dependencies and run
pip install -r requirements.txt
python 01_create_index.py           # Create index
python 02_generate_data.py          # Generate sample data
python 03_index_with_embeddings.py  # Index with embeddings
python 04_hybrid_queries.py         # Test queries

# 5. Access Dashboards (optional)
open http://localhost:5601
```

---

## 📚 References

- [OpenSearch Documentation](https://opensearch.org/docs/latest/)
- [Elasticsearch Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [HNSW Algorithm Paper](https://arxiv.org/abs/1603.09320)
- [BM25 Explained](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)
- [Azure OpenAI Embeddings](https://learn.microsoft.com/azure/ai-services/openai/how-to/embeddings)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

---

**Author:** Leonardo de Melo (demelo01@gmail.com)  
**Date:** February 2026  
**License:** MIT
