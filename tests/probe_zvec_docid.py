"""Probe which Zvec doc IDs the native library accepts. Read-only probe: it
creates a throwaway collection under data/cache_v2/_docid_probe and deletes
only that directory afterwards.  It never touches the canonical collections.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import zvec

PROBE = ROOT / "data" / "cache_v2" / "_docid_probe"
DIM = 8


def try_ids(ids):
    if PROBE.exists():
        shutil.rmtree(PROBE)
    schema = zvec.CollectionSchema(
        name="probe",
        fields=[zvec.FieldSchema("name", zvec.DataType.STRING)],
        vectors=[zvec.VectorSchema("text_vec", zvec.DataType.VECTOR_FP32,
                                   dimension=DIM,
                                   index_param=zvec.HnswIndexParam())],
    )
    col = zvec.create_and_open(str(PROBE), schema)
    vec = [0.1] * DIM
    for i in ids:
        try:
            col.insert([zvec.Doc(id=i, vectors={"text_vec": vec},
                                 fields={"name": i})])
            print(f"  OK    {len(i):>3}  {i!r}")
        except Exception as exc:
            print(f"  FAIL  {len(i):>3}  {i!r}  -> {str(exc)[:90]}")
    col.flush()
    col.close()


CASES = [
    "simple",
    "with_underscore",
    "with-hyphen",
    "with.pariod",
    "with:colon",
    "with,comma",
    "with(paren)",
    "with/slash",
    "with space",
    "MiXeD_Case-1",
    "a" * 32,
    "a" * 63,
    "a" * 64,
    "a" * 65,
    "a" * 100,
    "UPPER_CASE_" * 5,
    "1leading_digit",
    "0",
    "_leading_underscore",
    "-leading_hyphen",
    ".leading_dot",
    "9" * 64,
    "a" * 55 + "_" + "b" * 8,
]

print("zvec doc id probe")
try_ids(CASES)
shutil.rmtree(PROBE, ignore_errors=True)
print("probe dir removed")
