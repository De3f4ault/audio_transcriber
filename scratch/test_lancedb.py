import lancedb
import pyarrow as pa

db = lancedb.connect("./.lancedb")
schema = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("expression_id", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), 768)),
        pa.field("content", pa.string()),
        pa.field("source_type", pa.string()),
        pa.field("speaker", pa.string(), nullable=True),
        pa.field("created_at", pa.timestamp("us")),
    ]
)
if "test_hybrid2" in db.table_names():
    db.drop_table("test_hybrid2")
table = db.create_table("test_hybrid2", schema=schema)
table.add(
    [
        {
            "id": "1",
            "expression_id": 1,
            "vector": [0.0] * 768,
            "content": "dark magic horcrux",
            "source_type": "text",
            "created_at": None,
        }
    ]
)
table.create_fts_index("content")
import numpy as np

query_vector = np.random.rand(768).tolist()

try:
    results = table.search(query_type="hybrid").vector(query_vector).text("magic").to_list()
    print("search works", results)
except Exception as e:
    print(f"Error 2: {e}")

try:
    results = (
        table.search(query_type="hybrid", fts_columns="content")
        .vector(query_vector)
        .text("magic")
        .to_list()
    )
    print("search with fts_columns works", results)
except Exception as e:
    print(f"Error 3: {e}")
