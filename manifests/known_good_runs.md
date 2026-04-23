# Known-Good Runs

Append-only log. One entry per run ID that has been hand-reviewed and
blessed as reference material for future promotion. Latest entries at the
top.

Format:

```
## <run_id>
- checkpoint: <model_slug> (<family>)
- telemetry:  <source_label>
- heartbeat:  on|off
- saaq_rule:  saaq_v1_5 | legacy
- conclusion: <one line>
- artifacts:  <path under VALIDATION_OUTPUT_ROOT, or "artifacts/<run_id>/">
```

---

_No blessed runs yet. Bootstrap this file once Stage E is verified and the
first artifacts/ tree is reviewed._
