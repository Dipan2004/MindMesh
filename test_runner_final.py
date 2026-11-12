import sys,warnings,json,time
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0,"src"); sys.path.insert(0,"LightRAG")
sys.path.insert(0,"tests"); sys.path.insert(0,".")
import os; os.chdir("C:\\Users\\KIIT0001\\Downloads\\MindMesh-main\\MindMesh-main")

results = {}

def run(suite, fns):
    p,f,errs = [],[],{}
    for name,fn in fns:
        try: fn(); p.append(name)
        except Exception as e: f.append(name); errs[name]=str(e)[:120]
    results[suite]={"passed":p,"failed":f,"errors":errs}
    status = "OK  " if not f else "FAIL"
    print(f"[{status}] {suite}: {len(p)} pass / {len(f)} fail")
    for n in f: print(f"       FAIL {n}: {errs[n]}")

# Entity Resolution
import test_entity_resolution as ER
run("entity_resolution",[
  ("normalize_basic",ER.test_normalize_basic),("canonical_id",ER.test_canonical_id_deterministic),
  ("canonical_case",ER.test_canonical_id_case_insensitive),("canonical_types",ER.test_canonical_id_type_normalization),
  ("alias_builtin",ER.test_alias_resolve_builtin),("alias_keep",ER.test_alias_resolve_or_keep),
  ("alias_custom",ER.test_alias_register_custom),("type_compat",ER.test_type_compatible_same),
  ("type_incompat",ER.test_type_incompatible),("type_tech_concept",ER.test_type_tech_concept_compatible),
  ("candidates",ER.test_candidates_found),("no_cross",ER.test_same_doc_no_cross_doc),
  ("ambig_apple",ER.test_ambiguous_apple_not_merged),("ambig_java",ER.test_ambiguous_keeps_separate),
  ("merge_edges",ER.test_merge_edge_redirection),("merge_dedup",ER.test_merge_deduplicates_edges),
  ("no_selfloop",ER.test_merge_no_self_loops),("provenance",ER.test_merge_preserves_source_docs),
  ("descriptions",ER.test_merge_preserves_descriptions),("alias_loser",ER.test_merge_loser_becomes_alias),
  ("history",ER.test_merge_history_recorded),("rollback",ER.test_merge_rollback),
  ("cross_doc",ER.test_cross_doc_after_merge),("resolver",ER.test_resolver_configs),
  ("no_modify_orig",ER.test_resolver_does_not_modify_original),
])

# Pipeline Runner
import test_pipeline_runner as PR
run("pipeline_runner",[
  ("argparse_list",PR.test_argparse_list_pipelines),("argparse_models",PR.test_argparse_list_models),
  ("argparse_a_dry",PR.test_argparse_pipeline_a_dry_run),("argparse_p3",PR.test_argparse_pipeline3_hybrid),
  ("argparse_missing",PR.test_argparse_missing_model_requires_error),
  ("registry_all",PR.test_registry_all_pipelines_present),("registry_get",PR.test_registry_get_adapter_class),
  ("registry_unknown",PR.test_registry_unknown_pipeline_raises),
  ("model_valid",PR.test_model_validation_installed),("model_invalid",PR.test_model_validation_not_installed),
  ("adapter_a",PR.test_adapter_a_loads),("adapter_b",PR.test_adapter_b_loads),
  ("adapter_p3",PR.test_adapter_p3_loads_and_dry_runs),("lr_import",PR.test_lightrag_importable),
  ("lr_modes",PR.test_lightrag_queryparam_modes),("lr_version",PR.test_lightrag_version),
  ("p3_dry",PR.test_p3_dry_run_no_inference),("p3_unindexed",PR.test_p3_query_unindexed_returns_error),
  ("p3_workdir",PR.test_p3_workdir_isolated),("canonical_a",PR.test_pipeline_a_uses_canonical_not_experiment),
  ("cache_regression",PR.test_cache_isolation_regression),("no_overlap",PR.test_pipeline_registry_does_not_overlap_with_legacy),
])

# Cache Isolation
import test_model_cache_isolation as CI
run("cache_isolation",[
  ("diff_models",CI.test_different_models_produce_different_keys),
  ("same_settings",CI.test_same_model_same_settings_same_key),
  ("diff_penalty",CI.test_different_frequency_penalty_different_key),
])
# Tempdir tests
import tempfile; from pathlib import Path
def ci_extractor():
    with tempfile.TemporaryDirectory() as tmp:
        CI.test_extractor_uses_client_model_not_config_model(Path(tmp))
def ci_gleaning():
    with tempfile.TemporaryDirectory() as tmp:
        CI.test_gleaning_uses_client_model_not_config_model(Path(tmp))
run("cache_isolation_tempdir",[("extractor_model",ci_extractor),("gleaning_model",ci_gleaning)])

# Pipeline 3 CPU
import test_pipeline3_cpu as P3
run("pipeline3_cpu",[
  ("lr_import",P3.test_lightrag_import),("lr_ver",P3.test_lightrag_version),
  ("embed_func",P3.test_embedding_func_wrapper),("embed_adapter",P3.test_embedding_func_in_adapter),
  ("lr_init",P3.test_lightrag_init_dummy),("storage",P3.test_lightrag_storage_backends),
  ("workdir",P3.test_working_dir_isolation),("adapter_workdir",P3.test_adapter_workdir_isolated),
  ("qp_modes",P3.test_query_param_all_modes),("qp_ctx",P3.test_query_param_context_only),
  ("unindexed",P3.test_unindexed_returns_error),("dry_run",P3.test_dry_run_output),
  ("async_insert_query",P3.test_insert_and_query_dummy),
])

tp = sum(len(r["passed"]) for r in results.values())
tf = sum(len(r["failed"]) for r in results.values())
print(f"\nTOTAL: {tp} passed, {tf} failed")
with open("all_test_results.json","w",encoding="utf-8") as fh:
    json.dump(results,fh,indent=2)
print("Saved: all_test_results.json")
sys.exit(0 if tf==0 else 1)
