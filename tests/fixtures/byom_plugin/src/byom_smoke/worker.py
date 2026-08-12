"""Fresh-process reconstruction through slide2vec's ordinary model request."""

import json
import os
from pathlib import Path

from slide2vec import Model


request = json.loads(os.environ["BYOM_WORKER_REQUEST"])
model = Model.from_preset(
    request["name"],
    device="cpu",
    output_variant=request.get("output_variant"),
    allow_non_recommended_settings=bool(
        request.get("allow_non_recommended_settings", False)
    ),
)
Path(os.environ["BYOM_WORKER_SENTINEL"]).open("a").write(
    f"{model.name}:{model.feature_dim}\n"
)
