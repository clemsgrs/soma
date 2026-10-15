"""Device placement regressions for DenseModel."""

import torch

from soma.data.geometry import compute_dense_geometry
from soma.models.decoders.base import Decoder
from soma.models.heads.detection import DetectionHead
from soma.models.model import DenseModel
from soma.testing import FakeEncoder


def test_a_decoder_without_parameters_runs_where_the_model_was_moved():
    """The encoder stage returns its grid on its own device; a parameter-free
    decoder still runs where the model was moved, not where the grid came from."""

    class MeanDecoder(Decoder):
        def forward(self, X):
            return X.mean(dim=1, keepdim=True)

        @property
        def num_classes(self) -> int:
            return 1

    geometry = compute_dense_geometry(target_size=32, patch_size=4)
    head = DetectionHead(num_classes=1, geometry=geometry, spacing_um=1.0, match_distance=5.0, sigma=2.0)
    model = DenseModel(MeanDecoder(), head, encoder=FakeEncoder(target_size=32, patch_size=4)).to("meta")
    assert model.state_dict() == {}
    output = model(pixels=torch.zeros(1, 3, 32, 32, dtype=torch.uint8))
    assert output.logits.device.type == "meta"
