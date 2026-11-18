from graphrag_lite.retrieval.lightrag.chunk_lookup import fetch_chunks_by_ids

cids_d16 = ['417905a71f013c88', 'a09f4455b7fef7fa', 'f08417a79e9ed8e6', '6fc475866849cd38', '7bac4d8e30f7b55c', '4dcff615e7a2ceff']
chunks_map = fetch_chunks_by_ids(cids_d16)
print(f"Fetched {len(chunks_map)} of {len(cids_d16)} chunks for D16.")
for cid in cids_d16:
    c = chunks_map.get(cid, {})
    print(f"  {cid}: text length={len(c.get('text', ''))}, doc={c.get('source_doc')}")
