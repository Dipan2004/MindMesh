# Provider pilot notes (2026-11-20). Small timing and extraction-quality pilot, n = 2. Not a benchmark.

- The pilot script was run twice. The second run read every result from cache, so no extra API calls were
  made and nothing was extracted twice. Because of this, `fallback_events.jsonl` has two lines for the same
  self-test event. Since then, events carry a `from_cache` flag.
- The local qwen2.5:3b timings were measured with the laptop **on battery** (Win32_Battery status 1; GPU idle
  in P8 afterwards). Decode speed was about 49 tok/s, compared with about 98 tok/s measured earlier on AC power.
  The local times are slowed and should not be used as a comparison.
- Gemini's relationship agreement on 'New Mosque (Istanbul)' is 0 under exact-name matching. It names the
  mosque 'Yeni Cami', while Qwen and Mistral use 'New Mosque (Istanbul)'. Four of Qwen's five relations
  appear in Gemini's output under that name.
- 'Rango (2011 film)': all 3 Gemini attempts reached max_tokens = 4096 (finish_reason = length) and returned
  identical, truncated JSON. This is the same paragraph on which qwen3:4b failed. Retrying a deterministic
  request that stopped on length repeats the same output and costs money each time.
