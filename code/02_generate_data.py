"""
02_generate_data.py
===================

Gera dados fictícios de estabelecimentos para demonstração.
"""

import json
import random
from pathlib import Path

from faker import Faker

fake = Faker("pt_BR")
NUM_DOCUMENTS = 200

CNAES = [
    {"codigo": "0810-0/01", "descricao": "Extração de ardósia e beneficiamento associado"},
    {"codigo": "0810-0/02", "descricao": "Extração de granito e beneficiamento associado"},
    {"codigo": "0810-0/03", "descricao": "Extração de mármore e beneficiamento associado"},
    {"codigo": "0810-0/04", "descricao": "Extração de calcário e dolomita"},
    {"codigo": "0810-0/05", "descricao": "Extração de gesso e caulim"},
    {"codigo": "0810-0/06", "descricao": "Extração de areia, cascalho ou pedregulho"},
    {"codigo": "0810-0/07", "descricao": "Extração de argila e beneficiamento associado"},
    {"codigo": "2330-3/01", "descricao": "Fabricação de artefatos de concreto"},
    {"codigo": "2330-3/02", "descricao": "Fabricação de artefatos de cimento"},
    {"codigo": "2330-3/05", "descricao": "Preparação de massa de concreto e argamassa"},
    {"codigo": "2391-5/01", "descricao": "Britamento de pedras"},
    {"codigo": "2391-5/02", "descricao": "Aparelhamento de pedras para construção"},
    {"codigo": "4672-9/00", "descricao": "Comércio atacadista de ferragens e ferramentas"},
    {"codigo": "4679-6/02", "descricao": "Comércio atacadista de mármores e granitos"},
    {"codigo": "4679-6/04", "descricao": "Comércio atacadista de materiais de construção"},
    {"codigo": "4744-0/04", "descricao": "Comércio varejista de cal, areia, pedra e tijolos"},
    {"codigo": "4744-0/05", "descricao": "Comércio varejista de materiais de construção"},
    {"codigo": "4744-0/99", "descricao": "Comércio varejista de materiais de construção em geral"},
]

REGIOES = [
    {"cidade": "São Paulo", "uf": "SP", "lat": -23.5505, "lon": -46.6333, "raio": 0.5},
    {"cidade": "Rio de Janeiro", "uf": "RJ", "lat": -22.9068, "lon": -43.1729, "raio": 0.4},
    {"cidade": "Belo Horizonte", "uf": "MG", "lat": -19.9167, "lon": -43.9345, "raio": 0.3},
    {"cidade": "Curitiba", "uf": "PR", "lat": -25.4284, "lon": -49.2733, "raio": 0.3},
    {"cidade": "Porto Alegre", "uf": "RS", "lat": -30.0346, "lon": -51.2177, "raio": 0.3},
    {"cidade": "Salvador", "uf": "BA", "lat": -12.9714, "lon": -38.5014, "raio": 0.3},
    {"cidade": "Brasília", "uf": "DF", "lat": -15.8267, "lon": -47.9218, "raio": 0.4},
    {"cidade": "Fortaleza", "uf": "CE", "lat": -3.7172, "lon": -38.5433, "raio": 0.3},
    {"cidade": "Campinas", "uf": "SP", "lat": -22.9099, "lon": -47.0626, "raio": 0.2},
    {"cidade": "Goiânia", "uf": "GO", "lat": -16.6869, "lon": -49.2648, "raio": 0.3},
]

DESCRICOES = [
    "Empresa especializada em {atividade}. Atuamos há mais de {anos} anos no mercado.",
    "Fornecedor líder em {atividade}. Nossa empresa possui {funcionarios} funcionários.",
    "{atividade} com certificação ISO. Trabalhamos com os melhores materiais.",
    "Distribuidora de materiais para {atividade}. Entrega rápida e preços competitivos.",
    "Especialistas em {atividade}. Atendimento para obras de pequeno a grande porte.",
    "Comércio e beneficiamento de materiais para {atividade}. Frota própria.",
    "{atividade} desde {ano_fundacao}. Tradição e confiança no fornecimento.",
]


def gerar_cnpj() -> str:
    return f"{random.randint(10,99)}.{random.randint(100,999)}.{random.randint(100,999)}/0001-{random.randint(10,99)}"


def gerar_localizacao(regiao: dict) -> dict:
    lat = regiao["lat"] + random.uniform(-regiao["raio"], regiao["raio"])
    lon = regiao["lon"] + random.uniform(-regiao["raio"], regiao["raio"])
    return {"lat": round(lat, 6), "lon": round(lon, 6), "cidade": regiao["cidade"], "uf": regiao["uf"]}


def gerar_descricao(cnae: dict) -> str:
    template = random.choice(DESCRICOES)
    atividade = cnae["descricao"].lower().replace("comércio atacadista de ", "").replace("comércio varejista de ", "")
    return template.format(atividade=atividade, anos=random.randint(5, 40), funcionarios=random.randint(10, 500), ano_fundacao=random.randint(1980, 2020))


def gerar_documento() -> dict:
    cnae = random.choice(CNAES)
    regiao = random.choice(REGIOES)
    loc = gerar_localizacao(regiao)
    sufixos = ["Ltda", "ME", "EPP", "S.A.", "EIRELI"]
    tipos = ["Mineradora", "Comércio", "Distribuidora", "Indústria", "Materiais"]
    nome = f"{fake.last_name()} {random.choice(tipos)} {random.choice(sufixos)}"
    
    return {
        "_id": gerar_cnpj().replace(".", "").replace("/", "").replace("-", ""),
        "cnpj": gerar_cnpj(),
        "razao_social": nome.upper(),
        "nome_fantasia": nome.split()[0] + " " + random.choice(tipos),
        "cnae_codigo": cnae["codigo"],
        "cnae_descricao": cnae["descricao"],
        "descricao_atividade": gerar_descricao(cnae),
        "localizacao": {"lat": loc["lat"], "lon": loc["lon"]},
        "endereco": {
            "logradouro": fake.street_name(),
            "numero": str(random.randint(1, 9999)),
            "bairro": fake.bairro(),
            "cidade": loc["cidade"],
            "uf": loc["uf"],
            "cep": fake.postcode(),
        },
        "situacao_cadastral": random.choices(["ATIVA", "BAIXADA", "SUSPENSA"], weights=[0.85, 0.10, 0.05])[0],
        "capital_social": round(random.uniform(10000, 5000000), 2),
        "porte": random.choice(["MEI", "ME", "EPP", "DEMAIS"]),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("🏭 GERADOR DE DADOS FICTÍCIOS")
    print("=" * 60)
    
    documents = [gerar_documento() for _ in range(NUM_DOCUMENTS)]
    
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "estabelecimentos_demo.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ {len(documents)} documentos gerados")
    print(f"📁 Salvo em: {output_file}")
    
    cidades = {}
    for doc in documents:
        cidade = doc["endereco"]["cidade"]
        cidades[cidade] = cidades.get(cidade, 0) + 1
    
    print(f"\n📊 Estatísticas:")
    for cidade, count in sorted(cidades.items(), key=lambda x: -x[1])[:5]:
        print(f"   {cidade}: {count}")
