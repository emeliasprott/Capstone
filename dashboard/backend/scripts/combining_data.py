import polars as pl
from pathlib import Path

src = Path('dashboard/backend/data')

def concat_parts(pattern, out_name, src):
    pat = f"{src}/{pattern}"
    dfs = [pl.read_parquet(p) for p in src.glob(pattern)]
    pl.concat(dfs, how="vertical").write_parquet(src / out_name)

def main():
    concat_parts("*bills_kpis_*.parquet", "bill_kpis.parquet", src)
    concat_parts("*legislator_kpis_*.parquet", "legislator_kpis.parquet", src)
    concat_parts("*committee_kpis_*.parquet", "committee_kpis.parquet", src)
    concat_parts("*donor_kpis_*.parquet", "donor_kpis.parquet", src)
    concat_parts("*lobby_firm_kpis_*.parquet", "lobby_firm_kpis.parquet", src)
    concat_parts("*topic_snapshot_*.parquet", "topic_snapshot.parquet", src)
    for p in src.glob("*_embeddings_*.parquet"):
        p.rename(src / p.name)

if __name__ == "__main__":
    main()