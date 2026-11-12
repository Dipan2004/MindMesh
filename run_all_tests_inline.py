"""Run all CPU-safe tests and write results to JSON. No stdout issues."""
import sys, json, traceback, time, warnings, io, contextlib
warnings.filterwarnings("ignore")

ROOT_SRC = "src"
ROOT_LR  = "LightRAG"
ROOT_TESTS = "tests"
sys.path.insert(0, ROOT_SRC)
sys.path.insert(0, ROOT_LR)
sys.path.insert(0, ROOT_TESTS)

results = {}

def run_suite(suite_name, fn_list):
    suite = {"passed": [], "failed": [], "errors": {}}
    for fn_name, fn in fn_list:
        try:
            fn()
            suite["passed"].append(fn_name)
        except Exception as exc:
            suite["failed"].append(fn_name)
            suite["errors"][fn_name] = str(exc)
    results[suite_name] = suite
    return suite

# ── Suite 1: Entity Resolution ────────────────────────────────────────────────
import test_entity_resolution as ER
er_fns = [
    ("normalize_basic", ER.test_normalize_basic),
    ("normalize_unicode", ER.test_normalize_unicode),
    ("norm_alias_plurals", ER.test_norm_alias_plurals),
    ("ambiguous_detection", ER.test_ambiguous_detection),
    ("canonical_id_deterministic", ER.test_canonical_id_deterministic),
    ("canonical_id_case_insensitive", ER.test_canonical_id_case_insensitive),
    ("canonical_id_type_normalization", ER.test_canonical_id_type_normalization),
    ("canonical_id_different_names", ER.test_canonical_id_different_names),
    ("alias_resolve_builtin", ER.test_alias_resolve_builtin),
    ("alias_resolve_or_keep", ER.test_alias_resolve_or_keep),
    ("alias_register_custom", ER.test_alias_register_custom),
    ("type_compatible_same", ER.test_type_compatible_same),
    ("type_compatible_other", ER.test_type_compatible_other),
    ("type_incompatible", ER.test_type_incompatible),
    ("type_tech_concept_compatible", ER.test_type_tech_concept_compatible),
    ("candidates_found", ER.test_candidates_found),
    ("same_doc_no_cross_doc", ER.test_same_doc_no_cross_doc),
    ("ambiguous_apple_not_merged", ER.test_ambiguous_apple_not_merged),
    ("ambiguous_keeps_separate", ER.test_ambiguous_keeps_separate),
    ("merge_edge_redirection", ER.test_merge_edge_redirection),
    ("merge_deduplicates_edges", ER.test_merge_deduplicates_edges),
    ("merge_no_self_loops", ER.test_merge_no_self_loops),
    ("merge_preserves_source_docs", ER.test_merge_preserves_source_docs),
    ("merge_preserves_descriptions", ER.test_merge_preserves_descriptions),
    ("merge_loser_becomes_alias", ER.test_merge_loser_becomes_alias),
    ("merge_history_recorded", ER.test_merge_history_recorded),
    ("merge_rollback", ER.test_merge_rollback),
    ("cross_doc_after_merge", ER.test_cross_doc_after_merge),
    ("resolver_configs", ER.test_resolver_configs),
    ("resolver_does_not_modify_original", ER.test_resolver_does_not_modify_original),
]
run_suite("entity_resolution", er_fns)

# ── Suite 2: Pipeline Runner ──────────────────────────────────────────────────
sys.path.insert(0, ".")  # for run_pipeline
import test_pipeline_runner as PR
pr_fns = [
    ("argparse_list_pipelines", PR.test_argparse_list_pipelines),
    ("argparse_list_models", PR.test_argparse_list_models),
    ("argparse_pipeline_a_dry_run", PR.test_argparse_pipeline_a_dry_run),
    ("argparse_pipeline3_hybrid", PR.test_argparse_pipeline3_hybrid),
    ("argparse_missing_model", PR.test_argparse_missing_model_requires_error),
    ("registry_all_pipelines", PR.test_registry_all_pipelines_present),
    ("registry_get_class", PR.test_registry_get_adapter_class),
    ("registry_unknown_raises", PR.test_registry_unknown_pipeline_raises),
    ("model_validation_installed", PR.test_model_validation_installed),
    ("model_validation_not_installed", PR.test_model_validation_not_installed),
    ("adapter_a_loads", PR.test_adapter_a_loads),
    ("adapter_b_loads", PR.test_adapter_b_loads),
    ("adapter_p3_loads_dry_run", PR.test_adapter_p3_loads_and_dry_runs),
    ("lightrag_importable", PR.test_lightrag_importable),
    ("lightrag_queryparam_modes", PR.test_lightrag_queryparam_modes),
    ("lightrag_version", PR.test_lightrag_version),
    ("p3_dry_run_no_inference", PR.test_p3_dry_run_no_inference),
    ("p3_unindexed_returns_error", PR.test_p3_query_unindexed_returns_error),
    ("p3_workdir_isolated", PR.test_p3_workdir_isolated),
    ("pipeline_a_canonical", PR.test_pipeline_a_uses_canonical_not_experiment),
    ("cache_isolation_regression", PR.test_cache_isolation_regression),
    ("registry_no_overlap", PR.test_pipeline_registry_does_not_overlap_with_legacy),
]
run_suite("pipeline_runner", pr_fns)

# ── Suite 3: Cache Isolation ──────────────────────────────────────────────────
import test_model_cache_isolation as CI
ci_fns = [
    ("diff_models_diff_keys", CI.test_different_models_produce_different_keys),
    ("same_model_same_key", CI.test_same_model_same_settings_same_key),
    ("diff_penalty_diff_key", CI.test_different_frequency_penalty_different_key),
]
run_suite("cache_isolation", ci_fns)

# ── Suite 4: Pipeline 3 CPU ───────────────────────────────────────────────────
import test_pipeline3_cpu as P3
p3_non_async = [
    ("lightrag_import", P3.test_lightrag_import),
    ("lightrag_version", P3.test_lightrag_version),
    ("embedding_func_wrapper", P3.test_embedding_func_wrapper),
    ("embedding_func_in_adapter", P3.test_embedding_func_in_adapter),
    ("lightrag_init_dummy", P3.test_lightrag_init_dummy),
    ("storage_backends", P3.test_lightrag_storage_backends),
    ("workdir_isolation", P3.test_working_dir_isolation),
    ("adapter_workdir_isolated", P3.test_adapter_workdir_isolated),
    ("query_param_modes", P3.test_query_param_all_modes),
    ("query_param_context_only", P3.test_query_param_context_only),
    ("unindexed_returns_error", P3.test_unindexed_returns_error),
    ("dry_run_output", P3.test_dry_run_output),
]
# Async test
import asyncio
def run_async_test():
    P3.test_insert_and_query_dummy()
p3_all = p3_non_async + [("insert_and_query_dummy", run_async_test)]
run_suite("pipeline3_cpu", p3_all)

# ── Print and save ────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  ALL CPU-SAFE TEST RESULTS")
print("="*65)
total_pass = total_fail = 0
for suite_name, suite in results.items():
    p = len(suite["passed"])
    f = len(suite["failed"])
    total_pass += p
    total_fail += f
    status = "OK" if f == 0 else "FAIL"
    print(f"  [{status}] {suite_name:30} {p:3} pass  {f:3} fail")
    for fn in suite["failed"]:
        print(f"         FAIL: {fn}: {suite['errors'].get(fn,'')[:80]}")

print(f"\n  TOTAL: {total_pass} passed, {total_fail} failed")
print("="*65)

with open("all_test_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved: all_test_results.json")
sys.exit(0 if total_fail == 0 else 1)
