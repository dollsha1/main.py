# Databricks notebook source
# MAGIC %pip install mysql-connector-python
# MAGIC %pip install azure-search-documents
# MAGIC %pip install openai
# MAGIC %pip install python-dotenv
# MAGIC %pip install --upgrade azure-search-documents

# COMMAND ----------

# MAGIC %skip
# MAGIC %pip show azure-search-documents

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict

import mysql.connector
from mysql.connector import Error

from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
)

from openai import AzureOpenAI

# COMMAND ----------

MYSQL_CONFIG = {
    "host": "aecor-testdb.mysql.database.azure.com",
    "user": "mysqladmin",
    "password": dbutils.secrets.get("metadata-scope", "mysql-pass"),
    "database": "shop"
}

SEARCH_ENDPOINT = dbutils.secrets.get("metadata-scope", "search-endpoint")
SEARCH_KEY = dbutils.secrets.get("metadata-scope", "search-key")
SEARCH_INDEX_NAME = dbutils.secrets.get("metadata-scope", "search-index-name")

AOAI_ENDPOINT = dbutils.secrets.get("metadata-scope", "openai-endpoint")
AOAI_KEY = dbutils.secrets.get("metadata-scope", "openai-key")
model_name = "text-embedding-3-small"
deployment = "embedding-model"

EMBEDDING_CLIENT = AzureOpenAI(
    api_key=AOAI_KEY,
    api_version="2024-12-01-preview",
    azure_endpoint=AOAI_ENDPOINT,
)

# COMMAND ----------

def get_mysql_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)

# COMMAND ----------

def fetch_columns(cursor) -> List[Dict]:
    query = """
        SELECT
            table_schema,
            table_name,
            column_name,
            data_type,
            is_nullable,
            column_comment
        FROM information_schema.columns
        WHERE table_schema = %s
        ORDER BY table_name, ordinal_position
    """
    cursor.execute(query, (MYSQL_CONFIG["database"],))
    columns = cursor.fetchall()

    results = []
    for row in columns:
        results.append({
            "schema": row[0],
            "table": row[1],
            "column": row[2],
            "data_type": row[3],
            "nullable": row[4],
            "comment": row[5],
        })
    return results

# COMMAND ----------

def compute_profiling(cursor, schema: str, table: str, column: str) -> Dict:
    stats = {
        "null_ratio": None,
        "approx_distinct": None,
        "sample_values": [],
    }

    try:
        cursor.execute(f"SELECT COUNT(*) FROM `{schema}`.`{table}`")
        total = cursor.fetchone()[0]

        if total == 0:
            return stats

        cursor.execute(
            f"SELECT COUNT(*) FROM `{schema}`.`{table}` WHERE `{column}` IS NULL"
        )
        nulls = cursor.fetchone()[0]
        stats["null_ratio"] = round(nulls / total, 4)

        cursor.execute(
            f"SELECT COUNT(DISTINCT `{column}`) FROM `{schema}`.`{table}`"
        )
        stats["approx_distinct"] = cursor.fetchone()[0]

        cursor.execute(
            f"""
            SELECT `{column}`
            FROM `{schema}`.`{table}`
            WHERE `{column}` IS NOT NULL
            LIMIT 3
            """
        )
        stats["sample_values"] = [str(r[0]) for r in cursor.fetchall()]

    except Error:
        pass

    return stats

# COMMAND ----------

def build_text_blob(col: Dict, stats: Dict) -> str:
    return f"""
Table: {col['table']}
Column: {col['column']}
Type: {col['data_type']}
Nullable: {col['nullable']}
Comment: {col['comment'] or 'N/A'}
Null ratio: {stats['null_ratio']}
Approx distinct values: {stats['approx_distinct']}
Examples: {stats['sample_values']}
""".strip()

# COMMAND ----------

def generate_embedding(text: str) -> List[float]:
    response = EMBEDDING_CLIENT.embeddings.create(
        model=deployment,
        input=text,
    )
    return response.data[0].embedding

# COMMAND ----------

def create_search_index(embedding_dim: int):
    index_client = SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=AzureKeyCredential(SEARCH_KEY),
    )

    fields = [
        SearchField(name="id", type=SearchFieldDataType.String, key=True),
        SearchField(name="schema", type=SearchFieldDataType.String),
        SearchField(name="table", type=SearchFieldDataType.String),
        SearchField(name="column", type=SearchFieldDataType.String),
        SearchField(name="data_type", type=SearchFieldDataType.String),
        SearchField(name="nullable", type=SearchFieldDataType.String),
        SearchField(name="comment", type=SearchFieldDataType.String),
        SearchField(name="text_blob", type=SearchFieldDataType.String),
        SearchField(name="sample_values", type=SearchFieldDataType.String),
        SearchField(name="null_ratio", type=SearchFieldDataType.Double),
        SearchField(name="approx_distinct", type=SearchFieldDataType.Int64),
        SearchField(name="vector", type="Collection(Edm.Single)",
            vector_search_dimensions=embedding_dim,
            vector_search_profile_name="vector-profile",),
        SearchField(name="extracted_at", type=SearchFieldDataType.DateTimeOffset),
        SearchField(name="indexed_at", type=SearchFieldDataType.DateTimeOffset),
    ]

    vector_search = VectorSearch(
        algorithm_configurations=[
            VectorSearchAlgorithmConfiguration(
                name="default",
                kind="hnsw",
                parameters={"metric": "cosine"},
            )
        ]
    )

    index = SearchIndex(
        name=SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
    )

    try:
        index_client.create_or_update_index(index)
    except Exception:
        pass

# COMMAND ----------

def get_search_client():
    return SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_KEY),
        api_version = "2023-11-01"
    )

# COMMAND ----------

def extract_index():
    conn = get_mysql_connection()
    cursor = conn.cursor()

    columns = fetch_columns(cursor)
    documents = []

    embedding_dim = None

    for col in columns:
        stats = compute_profiling(
            cursor, col["schema"], col["table"], col["column"]
        )
        text_blob = build_text_blob(col, stats)
        embedding = generate_embedding(text_blob)

        if embedding_dim is None:
            embedding_dim = len(embedding)
            create_search_index(embedding_dim)

        doc = {
            "id": f"{col['column']}",
            "schema": col["schema"],
            "table": col["table"],
            "column": col["column"],
            "data_type": col["data_type"],
            "nullable": col["nullable"],
            "comment": col["comment"],
            "text_blob": text_blob,
            "sample_values": json.dumps(stats["sample_values"]),
            "null_ratio": stats["null_ratio"],
            "approx_distinct": stats["approx_distinct"],
            "vector": embedding,
            "extracted_at": datetime.utcnow(),
            "indexed_at": datetime.utcnow(),
        }
        documents.append(doc)

    search_client = get_search_client()
    search_client.upload_documents(documents)

    cursor.close()
    conn.close()

    print(f"Indexed {len(documents)} columns.")

# COMMAND ----------

extract_index()

# COMMAND ----------

def search(query: str):
    query_embedding = generate_embedding(query)

    search_client = get_search_client()

    results = search_client.search(
        search_text=None,
        vector_queries=[VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=5,
            fields="vector"
        )]
    )

    for r in results:
        print("-" * 80)
        print(f"Score: {r['@search.score']}")
        print(f"{r['schema']}.{r['table']}.{r['column']}")
        print(f"Type: {r['data_type']}")
        print(f"Comment: {r['comment']}")
        print(f"Samples: {r['sample_values']}")

# COMMAND ----------

query = "customer email address"
search(query)

# COMMAND ----------

search("order cancelled shipped status")