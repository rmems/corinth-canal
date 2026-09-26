# Known-Good Runs

Append-only log. One entry per run ID that has been hand-reviewed and
blessed as reference material for future promotion. Latest entries at the
top.

Format:

```
## <run_id>
- checkpoint: <model_slug> (<family>)
- telemetry:  <source_label>
- saaq_rule:  saaq_v1_5 | legacy
- conclusion: <one line>
- artifacts:  <path under VALIDATION_OUTPUT_ROOT, or "artifacts/<run_id>/">
```

---

## Spikenaut GPU dual-SAAQ CPU smoke — 2026-09-15 (RM-166)

- Model: `stub_olmoe` (`Olmoe`, `RoutingMode::StubUniform`)
- Family: `Olmoe`
- Rule: dual emit (`SaaqV1_5SqrtRate` primary + SAAQ 1.0 legacy columns)
- Telemetry: `csv_spikenaut_gpu`
- Domain: `spikenaut` (gpu / `neuromorphic_data.jsonl`)
- Rows: `4` (fixture `tests/fixtures/spikenaut/gpu_sample.jsonl`)
- Command: `just spikenaut-smoke tests/fixtures/spikenaut/gpu_sample.jsonl`
- Artifacts: `artifacts/spikenaut_gpu/dual_saaq_smoke/` (local smoke output; not tracked)

Conclusion: Spikenaut JSONL adapter produced canonical replay CSV; CPU dual-SAAQ smoke completed with `saaq_dual_emit: true` and domain tag `spikenaut_gpu`. Full GPU campaign replay of converted corpora is `TELEMETRY_SOURCE=csv TELEMETRY_CSV_PATH=… just saaq-csv`.

## SAAQ 1.5 OLMoE RE4 Control — 2026-04-23

NOTE (legacy control signal experiment, cleaned 2026-06 per GH#102): The heartbeat on/off entries below document null-result baselines from the old experimental control signal. `supports_heartbeat` + all related fields/columns/tick annotations + supporting code were removed. The artifact data directories were deleted in this hygiene pass (evidence of the null results is preserved in the text of this file and especially artifacts/issue-40-local/issue-40-local-summary.md). Current sviz profiles use clean condition-tagged runs only.

- Model: `olmoe_baseline`
- Family: `Olmoe`
- Rule: `SaaqV1_5SqrtRate`
- Telemetry: `csv_re4_path_tracing_telemetry`
- Heartbeat: `off`
- Repeat count: `2`
- Determinism: `matched`
- Rows: `2000`
- Run 0: `artifacts/olmoe_baseline/csv_re4_path_tracing_telemetry/heartbeat_off/20260423T195615_math_logic_r0_baseline_csv_off`
- Run 1: `artifacts/olmoe_baseline/csv_re4_path_tracing_telemetry/heartbeat_off/20260423T195637_math_logic_r1_baseline_csv_off`

Conclusion: heartbeat-off SAAQ 1.5 control baseline completed successfully on OLMoE with matched repeat determinism.

## SAAQ 1.5 OLMoE RE4 Baseline — 2026-04-23

- Model: `olmoe_baseline`
- Family: `Olmoe`
- Rule: `SaaqV1_5SqrtRate`
- Telemetry: `csv_re4_path_tracing_telemetry`
- Heartbeat: `on`
- Repeat count: `2`
- Determinism: `matched`
- Rows: `2000`
- Run 0: `artifacts/olmoe_baseline/csv_re4_path_tracing_telemetry/heartbeat_on/20260423T195816_math_logic_r0_baseline_csv_on`
- Run 1: `artifacts/olmoe_baseline/csv_re4_path_tracing_telemetry/heartbeat_on/20260423T195838_math_logic_r1_baseline_csv_on`

Conclusion: heartbeat-on SAAQ 1.5 validation completed successfully on OLMoE with matched repeat determinism.
