import glob, pickle, os
import faiss
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

in_dir = "dashboard/backend/data/"
pattern = "bill_embeddings_*.parquet"

paths = sorted(glob.glob(os.path.join(in_dir, pattern)))
assert paths, f"No parquet files matched {pattern}"

emb_list, id_list = [], []
for p in tqdm(paths, desc="Reading embeddings"):
    tbl = pq.read_table(p)
    emb_list.extend(tbl.column("embedding").to_pylist())
    id_list.extend(tbl.column("node_id").to_pylist())

emb = np.vstack(emb_list).astype("float32")
faiss.normalize_L2(emb)

index = faiss.index_factory(emb.shape[1], "HNSW32,Flat")
index.hnsw.efConstruction = 200
index.add(emb)

faiss_path = os.path.join(in_dir, f"bill_sim.faiss")
id_path = os.path.join(in_dir, f"bill_sim_ids.pkl")

faiss.write_index(index, faiss_path)
print(f"Saved index to {faiss_path}")
with open(id_path, "wb") as f:
    pickle.dump(id_list, f)
print(f"Saved IDs to {id_path}")