# Expanded web-only experiment

This snapshot contains the corrected dataset used for the second [solo training run](../../docs/solo-web-training.md). It reuses the permitted source cache and considers all 36 relevant candidate documents. It is not a fresh snapshot of every linked website.

- `corpus/`: 57 references and 598 chunks, with source URLs, authors, licenses, extraction changes and hashes.
- `export/`: 48 hypothetical cases from 16 accepted source families: 24 training, nine development, six calibration and nine test cases.
- `export/rejections.json`: 19 rejected candidate families. One additional family was quarantined before export.
- `audit.json`: an independent AI agent's integrity checks, coverage counts and limitations. This is not human expert validation.
- `repair-provenance.json`: the integrity corrections made before training or test predictions.
- [Measured results](../../docs/web-v2-results.json): one CPU epoch updating approximately 15M parameters in 322M-parameter Laya. Development accuracy stayed at **4/9 before and after**.

The first generated export contained copies of the same inputs in different splits, hidden by different citation IDs, and quotes consisting only of resource links. That original export was preserved locally and excluded from training. The validator now rejects both patterns. Three test source passages were replaced within their original source/check/split assignments; a weakly supported VisTrails family was quarantined. No model test predictions were used in these corrections. Test cases have not been scored.

| Split | Retry safety | Acknowledgement durability | Queue overload | Total |
|---|---:|---:|---:|---:|
| Training | 9 | 3 | 12 | 24 |
| Development | 3 | 0 | 6 | 9 |
| Calibration | 0 | 3 | 3 | 6 |
| Test | 0 | 6 | 3 | 9 |

Each represented decision has equal numbers of the three labels. Missing decisions and tiny samples prevent a reliable overall quality or calibration claim. The generated labels are checked against the same explicit rules used to construct cases; this is a consistency check, not independent ground truth. Exact duplicate checks cannot eliminate semantic overlap. Source concepts do not establish that a hypothetical messaging architecture survives its stated failures.

## Attribution and reuse

Source text and source-derived examples are **outside the MIT software grant**. AOSA chapter text remains CC BY 3.0; System Design Primer remains CC BY 4.0. Per-item authors, URLs, license links and extraction/adaptation changes are retained. See the [source license notices](../open-reference-v1/licenses/) and [third-party notices](../../THIRD_PARTY_NOTICES.md).

Cases and brief lessons are AI-authored adaptations with explicitly hypothetical mechanisms and requirements. Interview designs, including Twitter-style examples, are not verified descriptions of current company production systems. No human-reviewed decision dataset was created in this run.

## Reproduce the training run

From the repository root, with the model environment installed:

```powershell
.\.venv\Scripts\python.exe scripts\finetune_web.py --data datasets\web-solo-v2\export --base models\laya-multilingual --output models\web-head-v2-reproduction --epochs 1 --dry-run
.\.venv\Scripts\python.exe scripts\finetune_web.py --data datasets\web-solo-v2\export --base models\laya-multilingual --output models\web-head-v2-reproduction --epochs 1
```

Use your actual base checkpoint location. This run used revision `1720e3e3357cfe1e281542e223f8273b0890ca34`; the report records the weights hash. The approximately 60 MB experimental head requires that exact base checkpoint. It is not a standalone checkpoint and is not the plugin default. No model weights, credentials or private host logs are published here.
