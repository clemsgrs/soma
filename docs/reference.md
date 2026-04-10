# Reference

## Supported Encoders

| Model | Dim | Spacing |
|-------|-----|---------|
| uni / uni2 | 1024 / 1536 | 0.5 µm/px |
| virchow / virchow2 | 2560 | 0.5 µm/px |
| conch / conchv15 | 512 / 768 | 0.5 µm/px |
| gigapath | 1536 | 0.5 µm/px |
| h-optimus-0 / h-optimus-1 / h0-mini | 1536 | 0.5 µm/px |
| phikon / phikonv2 | 768 / 1024 | 0.5 µm/px |
| hibou-b / hibou-l | 768 / 1024 | 0.5 µm/px |
| midnight | 3072 | 0.25–2.0 µm/px |

## Attention Heatmap Config

`HeatmapConfig` controls whether and how attention heatmaps are generated after training. It is a field on `PipelineConfig`.

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` | Generate heatmaps after training completes. |
| `cmap` | `str` | `"coolwarm"` | Any matplotlib colormap name (e.g. `"viridis"`, `"hot"`, `"RdBu_r"`). |
| `alpha` | `float` | `0.5` | Opacity of the attention overlay blended onto the WSI thumbnail (0 = invisible, 1 = opaque). |
| `blur_sigma` | `float` | `0.0` | Gaussian blur standard deviation applied to the attention map before coloring. `0` disables blurring. |

The WSI thumbnail is read at the pyramid level closest to `preprocessing.seg_downsample` (default 64×), which matches the level used for tissue mask and tiling preview images. Heatmaps are saved to `fold_N/attention/` (raw scores) and `fold_N/heatmaps/` (PNG overlays) inside the run directory.

## Aggregators

Use the `aggregator.name` config key with the canonical lowercase names below.

| Aggregator | Config name | Attention heatmaps | Notes |
|---|---|---|---|
| `MeanPool` | `mean_pool` | — | Simple mean pooling baseline. |
| `MaxPool` | `max_pool` | — | Simple max pooling baseline. |
| `ABMIL` | `abmil` | ✓ | Attention-based MIL with gated attention. |
| `CLAM_SB` | `clam_sb` | ✓ | Reference-style single-branch CLAM with task-aware auxiliary supervision for classification, ordinal classification, and single-target regression. |
| `CLAM_MB` | `clam_mb` | ✓ (one map per class) | Reference-style multi-branch CLAM for classification tasks only. |

### CLAM task support

`clam_sb` adapts its auxiliary loss to the task head it is paired with:

| Task family | Auxiliary behavior |
|---|---|
| `classification` | Original CLAM-style instance clustering with per-class instance classifiers. |
| `ordinal_classification` | Top-k scalar instance regression toward the bag label plus low-attention regularization. |
| `regression` | Top-k scalar instance regression toward the bag target plus low-attention regularization. |

Key `clam_sb` parameters:
- `bag_weight`: mixes bag loss and auxiliary loss as `bag_weight * task_loss + (1 - bag_weight) * auxiliary_loss`
- `instance_loss_mode`: optional explicit mode override; defaults to the paired task family and must agree with it
- `low_attention_weight`: weight on the low-attention regularization term for ordinal/regression CLAM
- `topk_target_weight`: weight on the top-k target-matching term for ordinal/regression CLAM
- `use_negative_class_instance_loss`: classification-only flag for training out-of-class negative instance branches

`clam_mb` remains class-branch-specific and is therefore limited to categorical classification.
| `DSMIL` | `dsmil` | ✓ | Dual-stream MIL with critical-instance attention. |
| `TransMIL` | `transmil` | — | Transformer-based MIL with Nyström attention. |
| `DTFDMIL` | `dtfdmil` | — | Double-tier feature distillation MIL. |
| `HIPT` | `hipt` | — | Hierarchical MIL over native hierarchical features. |

## Task Heads

Task heads map the aggregated slide representation to a prediction and define the training loss, label encoding, and evaluation metrics.

### `classification`

Linear head for binary or multi-class classification.

| Property | Value |
|---|---|
| Loss | Cross-entropy |
| Label dtype | `torch.long` (integer class indices) |
| Metrics | `accuracy`, `balanced_accuracy`, `f1_macro`, `auc` |
| Auto-injected param | `num_classes` (from dataset) |

Config:
```yaml
task:
  name: classification
```

User-provided `params` are merged on top of the auto-injected `num_classes`, so no extra config is required for standard use.

### `branch_aware_classification`

Classification head for branch-aware MIL representations shaped `(B, C, D)`.
This is primarily intended for `clam_mb`, which emits one pooled representation
per class branch.

| Property | Value |
|---|---|
| Loss | Cross-entropy |
| Label dtype | `torch.long` (integer class indices) |
| Metrics | `accuracy`, `balanced_accuracy`, `f1_macro`, `auc` |
| Auto-injected param | `num_classes` (from dataset) |

Config:
```yaml
task:
  name: branch_aware_classification
```

### `ordinal_classification`

Linear head for ordered integer labels (e.g. grading scores 0–5). Uses MSE loss during training — treating labels as continuous values — then rounds and clips the continuous output to the nearest integer class at inference. Both the rounded integer prediction and the raw continuous score are saved.

| Property | Value |
|---|---|
| Loss | MSE (targets cast to float) |
| Label dtype | `torch.long` (integer class indices) |
| Metrics | `qwk` (quadratic weighted kappa), `accuracy`, `balanced_accuracy` |
| Auto-injected param | `num_classes` (from dataset, used for clipping) |

Config:
```yaml
task:
  name: ordinal_classification
```

`predictions.csv` columns: `sample_id`, `true_label`, `predicted_label`, `raw_score`.

`clam_sb` is compatible with `ordinal_classification`. In that pairing, CLAM keeps its attention-based top-k selection but replaces class-clustering with scalar instance regression toward the bag’s ordinal label.

### `regression`

Linear head for single or multi-target regression.

| Property | Value |
|---|---|
| Loss | Mean squared error (MSE) |
| Label dtype | `torch.float` (continuous values) |
| Metrics | `mse`, `mae`, `r2` |
| Auto-injected param | none |

Config:
```yaml
task:
  name: regression
  params:
    num_targets: 1   # optional, defaults to 1
```

Labels in `dataset.csv` must be numeric floats. No label encoding is applied — values are used as-is.

`clam_sb` is compatible with single-target `regression`. Multi-target regression is not currently supported with CLAM auxiliary supervision.

### Adding a custom task head

Subclass `TaskHead` and register it:

```python
import torch
import torch.nn.functional as F
from soma.tasks.base import TaskHead
from soma.tasks.registry import task_registry

class MyHead(TaskHead):
    label_dtype = torch.long

    def __init__(self, input_dim: int, **kwargs) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, 1)

    @classmethod
    def auto_params(cls, dataset):
        return {}   # inject dataset-derived kwargs here if needed

    def forward(self, X):
        return self.fc(X)

    def compute_loss(self, predictions, targets):
        return F.binary_cross_entropy_with_logits(predictions.squeeze(-1), targets.float())

    def postprocess(self, raw_output):
        return {"predictions": raw_output.squeeze(-1).detach().cpu().numpy()}

    def compute_metrics(self, raw_output, targets):
        return {}

task_registry.register("my_head", MyHead)
```

Then use `task: {name: my_head}` in the pipeline config.
