# Reference

## Supported Encoders

The canonical encoder presets are registered in code and summarized below. Use the tables as the source of truth for:

- which encoder names ship with `soma`
- whether a preset is a tile encoder or a slide encoder
- the output dimension returned by each preset
- which spacing values are accepted by validation
- preset-specific behavior, including `output_variant` support

### Tile-level encoders (15)

| Preset | Model | Output dim | Supported spacing | Notes |
| --- | --- | --- | --- | --- |
| `uni` | [UNI](https://huggingface.co/MahmoodLab/UNI) | 1024 | `0.5` µm/px | |
| `uni2` | [UNI2](https://huggingface.co/MahmoodLab/UNI2-h) | 1536 | `0.5` µm/px | |
| `virchow` | [Virchow](https://huggingface.co/paige-ai/Virchow) | 1280 / 2560 | `0.5` µm/px | Supports `output_variant="cls"` or `"cls_patch_mean"` |
| `virchow2` | [Virchow2](https://huggingface.co/paige-ai/Virchow2) | 1280 / 2560 | `0.5`, `1.0`, `2.0` µm/px | Supports `output_variant="cls"` or `"cls_patch_mean"` |
| `conch` | [CONCH](https://huggingface.co/MahmoodLab/conch) | 512 | `0.5` µm/px | |
| `conchv15` | [CONCHv1.5](https://huggingface.co/MahmoodLab/TITAN) | 768 | `0.5` µm/px | |
| `gigapath` | [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) | 1536 | `0.5` µm/px | Alias: `prov-gigapath` |
| `h-optimus-0` | [H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) | 1536 | `0.5` µm/px | |
| `h-optimus-1` | [H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) | 1536 | `0.5` µm/px | |
| `h0-mini` | [H0-mini](https://huggingface.co/bioptimus/H0-mini) | 768 / 1536 | `0.5` µm/px | Supports `output_variant="cls"` or `"cls_patch_mean"` |
| `phikon` | [Phikon](https://huggingface.co/owkin/phikon) | 768 | `0.5` µm/px | |
| `phikonv2` | [Phikon-v2](https://huggingface.co/owkin/phikon-v2) | 1024 | `0.5` µm/px | |
| `hibou-b` | [Hibou-B](https://huggingface.co/histai/hibou-b) | 768 | `0.5` µm/px | |
| `hibou-l` | [Hibou-L](https://huggingface.co/histai/hibou-L) | 1024 | `0.5` µm/px | |
| `midnight` | [MidNight12k](https://huggingface.co/kaiko-ai/midnight) | 3072 | `0.25`, `0.5`, `1.0`, `2.0` µm/px | Alias: `kaiko-midnight` |

### Slide-level encoders (3)

| Preset | Model | Tile encoder | Output dim | Supported spacing | Notes |
| --- | --- | --- | --- | --- | --- |
| `gigapath-slide` | [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) | `gigapath` | 768 | `0.5` µm/px | |
| `prism` | [PRISM](https://huggingface.co/paige-ai/PRISM) | `virchow` (`cls_patch_mean`) | 1280 | `0.5` µm/px | |
| `titan` | [TITAN](https://huggingface.co/MahmoodLab/TITAN) | `conchv15` | 768 | `0.5` µm/px | |

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
| `binary_classification` | Original CLAM-style instance clustering with per-class instance classifiers. |
| `multiclass_classification` | Original CLAM-style instance clustering with per-class instance classifiers. |
| `ordinal_classification` | Top-k scalar instance regression toward the bag label plus low-attention regularization. |
| `regression` | Top-k scalar instance regression toward the bag target plus low-attention regularization. |

Key `clam_sb` parameters:
- `bag_weight`: mixes bag loss and auxiliary loss as `bag_weight * task_loss + (1 - bag_weight) * auxiliary_loss`
- `instance_loss_mode`: optional explicit mode override; defaults to the paired task family and must agree with it
- `low_attention_weight`: weight on the low-attention regularization term for ordinal/regression CLAM
- `topk_target_weight`: weight on the top-k target-matching term for ordinal/regression CLAM
- `use_negative_class_instance_loss`: classification-only flag for training out-of-class negative instance branches

`clam_mb` remains class-branch-specific and is therefore limited to `multiclass_classification` (and `branch_aware_classification`).
| `DSMIL` | `dsmil` | ✓ | Dual-stream MIL with critical-instance attention. |
| `TransMIL` | `transmil` | — | Transformer-based MIL with Nyström attention. |
| `DTFDMIL` | `dtfdmil` | — | Double-tier feature distillation MIL. |
| `HIPT` | `hipt` | — | Hierarchical MIL over native hierarchical features. |

## Task Heads

Task heads map the aggregated slide representation to a prediction and define the training loss, label encoding, and evaluation metrics.

### Configuring metrics

Every task head accepts an optional `metrics` list in the config. An empty list (the default) uses the task’s built-in defaults. Specify metric names explicitly to override:

```yaml
task:
  name: binary_classification
  metrics: [auroc, sensitivity, specificity]
```

The set of valid metrics differs per task family — see each section below.

### `binary_classification`

Linear head for two-class classification (`num_classes` must equal 2).

| Property | Value |
|---|---|
| Loss | Cross-entropy |
| Label dtype | `torch.long` (integer class indices) |
| Default metrics | `auroc`, `balanced_accuracy`, `auprc`, `f1` |
| All valid metrics | `accuracy`, `balanced_accuracy`, `auroc`, `auprc`, `f1`, `sensitivity`, `specificity`, `mcc` |
| Auto-injected param | `num_classes` (from dataset) |

Config:
```yaml
task:
  name: binary_classification
```

### `multiclass_classification`

Linear head for classification with three or more classes.

| Property | Value |
|---|---|
| Loss | Cross-entropy |
| Label dtype | `torch.long` (integer class indices) |
| Default metrics | `auroc_macro`, `balanced_accuracy`, `f1_macro` |
| All valid metrics | `accuracy`, `balanced_accuracy`, `auroc_macro`, `f1_macro`, `f1_weighted`, `mcc` |
| Auto-injected param | `num_classes` (from dataset) |

Config:
```yaml
task:
  name: multiclass_classification
```

### `branch_aware_classification`

Classification head for branch-aware MIL representations shaped `(B, C, D)`.
This is primarily intended for `clam_mb`, which emits one pooled representation
per class branch. Accepts the same metrics as `multiclass_classification`.

| Property | Value |
|---|---|
| Loss | Cross-entropy |
| Label dtype | `torch.long` (integer class indices) |
| Default metrics | `auroc_macro`, `balanced_accuracy`, `f1_macro` |
| All valid metrics | `accuracy`, `balanced_accuracy`, `auroc_macro`, `f1_macro`, `f1_weighted`, `mcc` |
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
| Default metrics | `qwk`, `balanced_accuracy` |
| All valid metrics | `qwk`, `linear_wk`, `accuracy`, `balanced_accuracy`, `mae`, `spearman` |
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
| Default metrics | `mae`, `r2` |
| All valid metrics | `mse`, `rmse`, `mae`, `r2`, `pearson`, `spearman` |
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
