# Deviations and incidents: expanded HotpotQA experiments

| When | Event | Decision / handling |
|---|---|---|
| 2026-11-19 20:07–20:56 | Main 100q run paused for the extraction-variant validation (user-approved). 33 records on disk. | Resumed unchanged at 20:56 (mistral:latest, original prompt). |
| 2026-11-19 20:40 | Thermal hard reboot during the validation. | Validation resumed; main run was paused at the time. |
| 2026-11-19 20:42 | The Startup auto-resume launcher would have restarted the main run at logon. | Disabled (`MindMesh_HotpotQA_resume.cmd.paused`) and the pending launcher stopped. Stays disabled per user. |
| 2026-11-19 20:56 | `HOLD_200Q` added. | The driver stops after the 100q stage; 200q needs explicit approval. |
| 2026-11-19 ~21:00–21:35 | 18 forced thermal sleep/resume cycles (136 critical-thermal events); no reboot. | The run continued; processes survived. |
| 2026-11-19 (100q extraction) | Paragraph **'Writers of the Future'** (`1dd9dfe094f24a13`, a distractor for question `5a7542425542996c70cfaecc`, not gold evidence) failed after 305 s with `ResponseError: CUDA error: unknown error (status 500)`. This is a GPU fault, most likely from a forced thermal sleep during inference, not a model output failure. | **User decision: record as a failure; do not re-extract.** It stays in the dense index and contributes nothing to the graph. Reported as an *infrastructure (GPU/thermal) extraction failure*, separately from model parse failures. |
| 2026-11-19 21:40–21:41 | **The full-corpus 100q extraction was stopped by explicit user request.** The driver (PID 14308) was stopped first, at 21:40:20 with 103 records. The extractor (PID 1828) was stopped at 21:41:02, immediately after it flushed record 104 ('Divya Rana'), so no paragraph was cut off mid-write. | Checkpoint preserved: 104 of 985 records (102 clean, 2 failures); 881 paragraphs not extracted. No automatic restart is possible: the driver is stopped, the Startup launcher is disabled, and HOLD_200Q is present. `status.json` was left unmodified and still reads "running". |
| 2026-11-19 (100q extraction) | Paragraph 'Istiklal Mosque' (`1cbf4009842f5c2b`, distractor) failed the retry ladder with malformed JSON. | Normal model extraction failure; counted as such. |
