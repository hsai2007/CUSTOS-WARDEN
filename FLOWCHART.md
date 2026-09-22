# WARDEN Decision and Evaluation Flow

```mermaid
flowchart TD
    A1["M7 · LangGraph agents<br/>demo only · feeds no metric"] --> GW
    A2["M5 · workload.py<br/>labelled writes · all metrics"] --> HN

    GW["<b>gateway.py</b><br/>public API · submit(key, text)"]
    HN["<b>harness.py</b><br/>evaluation entry point"]

    GW --> RD
    HN --> RD

    RD["1 · read current value + version<br/>store.py"]
    RD --> BOOT{"new key?"}
    BOOT -->|yes · bootstrap| APPLY
    BOOT -->|no| FP

    FP["2 · FAST PATH · 92.9% of writes · 0.12 ms<br/>classifier.py + facts.py · regex only, no embeddings"]
    FP --> FC{"facts compatible?<br/>date · number · negation"}

    FC -->|yes| DET{"adds detail?<br/>trailing clause / extra facts"}
    DET -->|yes| REF["refinement"]
    DET -->|no| DUP["duplicate"]

    FC -->|no| MK{"supersession<br/>marker present?"}
    MK -->|no| CON1["contradiction<br/>caught cheaply"]
    MK -->|yes| ESC

    ESC["3 · SLOW PATH · 7.1% of writes · 1.6 s<br/>LLM oracle · qwen/qwen3.8-27b<br/>real supersession, or a conflict in update clothing?"]
    ESC -->|UPDATE| UPD["update"]
    ESC -->|CONTRADICTION| CON2["contradiction<br/>caught by the model"]

    DUP --> DROP["drop"]
    REF --> MERGE["merge"]
    UPD --> APPLY["apply"]
    CON1 --> BLOCK["block"]
    CON2 --> BLOCK

    DROP --> GATE
    MERGE --> GATE
    APPLY --> GATE
    BLOCK --> GATE

    GATE["4 · gate.py<br/>maps class to action"]
    GATE --> ST[("M1 · store.py<br/>value · version · full history")]
    GATE --> LOG[/"writes.csv · 12 frozen columns<br/>ts · config · write_id · key · true_class · predicted_class<br/>action · is_false · version_read · version_at_commit<br/>classify_ms · escalated"/]
    LOG --> MET["5 · metrics.py<br/>accuracy · F1 · survival · cost<br/>reads the log only, never re-simulates"]
    MET --> OUT["M6 · generate_results.py<br/>9 output artefacts"]

    subgraph SIDE ["outside the WARDEN decision path"]
        direction TB
        GT["Ground truth · true_class + is_false<br/>owned by M5, never visible to the classifier"]
        ABL["CosineOnlyClassifier · all-MiniLM-L6-v2<br/>ABLATION BASELINE<br/>30.8% accuracy · 19.9 ms per write"]
    end
    GT -.-> A2

    classDef fast fill:#EDF1F6,stroke:#5A6B7F,color:#1B2430
    classDef slow fill:#F2E7DE,stroke:#C2703D,color:#1B2430
    classDef act fill:#1B2430,stroke:#1B2430,color:#FFFFFF
    classDef store fill:#DDE4EC,stroke:#5A6B7F,color:#1B2430
    classDef entry fill:#FFFFFF,stroke:#5A6B7F,color:#1B2430
    classDef abl fill:#F7F7F7,stroke:#AAAAAA,color:#666666,stroke-dasharray: 4 3
    classDef truth fill:#F6E3E3,stroke:#B05A5A,color:#1B2430

    class FP,FC,DET,MK,BOOT,DUP,REF,CON1,UPD fast
    class ESC,CON2 slow
    class DROP,MERGE,APPLY,BLOCK,GATE act
    class RD,ST,LOG,MET,OUT store
    class GW,HN,A1,A2 entry
    class ABL abl
    class GT truth
    style SIDE fill:#FFFFFF,stroke:#CCCCCC,stroke-dasharray: 5 4,color:#888888
```

## Implementation notes

- `gate.py` maps a classifier result to an action. The actual state change is performed by `Store.write()` in `store.py`; bootstrap writes are written directly by `gateway.py` or `harness.py`.
- `CosineOnlyClassifier` is an ablation baseline. The production `FastPathEscalator` in `classifier.py` uses extracted facts and supersession-marker rules before optional LLM escalation; it does not call the embedding model on its fast path.
- The percentage and latency labels reflect the published `results/metrics_table.csv` WARDEN run.
