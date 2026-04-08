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

## Aggregators

- `ABMIL` - attention-based MIL with gated attention.
- `MeanPool` - simple mean pooling baseline.
- `MaxPool` - simple max pooling baseline.

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
