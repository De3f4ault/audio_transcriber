from lancedb.rerankers import ColbertReranker

# Monkey-patch for transformers compatibility
from rerankers.models.colbert_ranker import ColBERTModel

if not hasattr(ColBERTModel, "all_tied_weights_keys"):
    ColBERTModel.all_tied_weights_keys = {}

print("Loading ColbertReranker...")
reranker = ColbertReranker(model_name="answerdotai/answerai-colbert-small-v1")
print("Loaded!")

# Check if we can just rerank raw text
import lancedb

# Create an in-memory LanceDB table
db = lancedb.connect("memory://")
data = [
    {"id": 1, "text": "The magic system in Harry Potter uses wands."},
    {"id": 2, "text": "Horcruxes are created through murder and dark magic."},
    {"id": 3, "text": "The Force is an energy field created by all living things."},
]
tbl = db.create_table("test", data=data)

# Rerank with Colbert
tbl.create_fts_index("text", replace=True)
res = tbl.search("dark magic horcrux").rerank(reranker=reranker).limit(3).to_pandas()
print("Colbert Reranker Results:")
print(res)
