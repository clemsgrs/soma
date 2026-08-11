"""Tests for the reusable Croma 0.3 tile-encoder panel audit."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from typing import Any, cast

import pytest
from slide2vec.encoders import encoder_registry

from soma.benchmarks.croma import (
    CROMA_0_3_ENCODER_PANEL,
    validate_croma_0_3_encoder_panel,
)

_AUDIT_FIXTURE = Path(__file__).parent / "fixtures" / "croma-0.3-encoder-audit.json"


def _audit_fixture() -> dict[str, Any]:
    return json.loads(_AUDIT_FIXTURE.read_text(encoding="utf-8"))


def test_croma_0_3_encoder_panel_matches_the_published_roster() -> None:
    fixture = _audit_fixture()
    expected = {
        row["published_model"]: (
            row["expected_soma_encoder"],
            row["expected_output_variant"],
            row["expected_dimension"],
        )
        for row in fixture["encoder_audit"]
    }

    observed = {
        published_name: (
            spec.soma_encoder,
            spec.output_variant,
            spec.dimension,
        )
        for published_name, spec in CROMA_0_3_ENCODER_PANEL.items()
    }

    assert len(CROMA_0_3_ENCODER_PANEL) == 26
    assert len({spec.soma_encoder for spec in CROMA_0_3_ENCODER_PANEL.values()}) == 26
    assert observed == expected
    assert fixture["source"]["release"] == "0.3.0"
    assert all(
        row["croma_dimension"] == row["expected_dimension"]
        for row in fixture["encoder_audit"]
    )


def test_croma_0_3_encoder_panel_is_immutable() -> None:
    panel = cast(Any, CROMA_0_3_ENCODER_PANEL)

    with pytest.raises(TypeError):
        panel["CONCH"] = panel["Virchow2"]
    with pytest.raises(FrozenInstanceError):
        panel["CONCH"].dimension = 999


def test_croma_0_3_encoder_panel_translates_all_234_reference_rows() -> None:
    reference_rows = _audit_fixture()["reference_rows"]

    soma_encoder_keys = [
        CROMA_0_3_ENCODER_PANEL[row["published_model"]].soma_encoder
        for row in reference_rows
    ]
    expected_encoder_keys = [row["expected_soma_encoder"] for row in reference_rows]

    assert len(soma_encoder_keys) == 234
    assert soma_encoder_keys == expected_encoder_keys


def test_croma_0_3_encoder_panel_matches_slide2vec_registry_metadata() -> None:
    validate_croma_0_3_encoder_panel()


@pytest.fixture
def registry_metadata() -> dict[str, dict[str, Any]]:
    return {
        spec.soma_encoder: encoder_registry.info(spec.soma_encoder)
        for spec in CROMA_0_3_ENCODER_PANEL.values()
    }


def _assert_drift_context(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "encoder='conch'" in message
    assert "requested_variant='default'" in message
    assert "expected_dimension=512" in message
    assert "observed_metadata=" in message


def test_croma_0_3_encoder_panel_reports_dimension_metadata_drift(
    registry_metadata: dict[str, dict[str, Any]],
) -> None:
    metadata = registry_metadata
    metadata["conch"] = {
        **metadata["conch"],
        "output_variants": {"default": {"encode_dim": 513}},
    }

    with pytest.raises(ValueError) as error:
        validate_croma_0_3_encoder_panel(metadata_by_encoder=metadata)

    _assert_drift_context(error)
    assert "'encode_dim': 513" in str(error.value)


def test_croma_0_3_encoder_panel_does_not_approximate_a_missing_variant(
    registry_metadata: dict[str, dict[str, Any]],
) -> None:
    metadata = registry_metadata
    metadata["conch"] = {
        **metadata["conch"],
        "default_output_variant": "replacement",
        "output_variants": {"replacement": {"encode_dim": 512}},
    }

    with pytest.raises(ValueError) as error:
        validate_croma_0_3_encoder_panel(metadata_by_encoder=metadata)

    _assert_drift_context(error)
    assert "'replacement': {'encode_dim': 512}" in str(error.value)
