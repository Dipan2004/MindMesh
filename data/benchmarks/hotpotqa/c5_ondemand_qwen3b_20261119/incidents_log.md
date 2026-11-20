# Incidents: Dense vs on-demand C5 (qwen2.5:3b), run `dense_vs_c5_6ca3b11e`

| When | Event | Handling |
|---|---|---|
| 2026-11-19 22:39 | Full run started (PID 10592) with a cold cache. | — |
| 2026-11-19 22:49–22:51 | Zvec keyword (BM25) parser warnings: `fts syntax error ... 'and'` and `field-prefixed queries are not supported`. | Per user instruction: not stopped, not modified. The existing fallback (vector-only query) applies; counted in the final report. |
| 2026-11-19 23:04:21 | **Thermal hard reboot.** The run stopped in question 38. | Checkpoint verified before restart: 75 retrieval rows (Dense q1–38, C5 q1–37), 0 generation rows, 197 cache entries (195 ok, 2 failed_model). |
| (same) | Question 38's C5 step had already saved 2 extractions ('Bonifacio High Street' 13.19 s, 'High Street (IND Eighth Avenue Line)' 10.56 s). A third, in-progress extraction was lost unsaved and will be redone. | On resume these 2 are cache hits for q38, so q38's per-question extraction time understates by 23.75 s. Total measured extraction cost is computed from cache entries (by `triggered_by_question`), so it includes them. The redone partial extraction is not double-counted (the lost attempt left no record). |
| 2026-11-20 | Resumed with the identical command by user request (auto-resume stays disabled). | Finished rows are skipped; nothing is rerun. |
