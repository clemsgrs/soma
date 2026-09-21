"""The class scheme shared by segmentation (raster values) and detection (point ids)."""

from __future__ import annotations

import pytest

from soma.class_scheme import assign_raw_values, resolve_classes


def test_assign_maps_each_raw_value_to_its_class_index_in_declaration_order():
    class_of, excluded = assign_raw_values(
        {"mnl": [0, 1], "other": 7}, excluded=[2], excluded_name="drop"
    )

    assert class_of == {0: 0, 1: 0, 7: 1}
    assert excluded == [2]


def test_assign_accepts_values_above_255_without_a_ceiling():
    class_of, _ = assign_raw_values({"cell": [300]}, excluded=(), excluded_name="drop")

    assert class_of == {300: 0}


def test_assign_rejects_values_above_the_ceiling():
    with pytest.raises(ValueError, match=r"integers in \[0, 255\], got 300"):
        assign_raw_values({"cell": [300]}, excluded=(), excluded_name="ignore", max_value=255)


@pytest.mark.parametrize("value", [-1, 1.5, True, "1"])
def test_assign_rejects_non_integer_or_negative_values(value):
    with pytest.raises(ValueError, match="non-negative integers"):
        assign_raw_values({"cell": [value]}, excluded=(), excluded_name="drop")


def test_assign_rejects_a_value_in_a_class_and_in_the_excluded_list():
    with pytest.raises(ValueError, match="'cell' and 'drop' both list raw value 1"):
        assign_raw_values({"cell": [0, 1]}, excluded=[1], excluded_name="drop")


def test_resolve_derives_num_classes_and_names_from_classes():
    resolved = resolve_classes(
        {"classes": {"mnl": [0, 1]}, "drop": [2]}, excluded_key="drop", subject="detection"
    )

    assert resolved == (1, ("mnl",), {0: 0, 1: 0}, (2,))


def test_resolve_with_num_classes_alone_keeps_ids_as_they_are():
    resolved = resolve_classes({"num_classes": 2}, excluded_key="drop", subject="detection")

    assert resolved == (2, ("class_0", "class_1"), None, ())


def test_resolve_rejects_num_classes_disagreeing_with_classes():
    with pytest.raises(ValueError, match="num_classes=3 disagrees with the 1 classes"):
        resolve_classes(
            {"num_classes": 3, "classes": {"mnl": [0, 1]}},
            excluded_key="drop",
            subject="detection",
        )


def test_resolve_rejects_excluded_key_without_classes():
    with pytest.raises(ValueError, match="task.params.drop needs task.params.classes"):
        resolve_classes({"num_classes": 2, "drop": [2]}, excluded_key="drop", subject="detection")


def test_resolve_requires_classes_or_num_classes():
    with pytest.raises(ValueError, match="dataset_type='detection' requires task.params.classes"):
        resolve_classes({}, excluded_key="drop", subject="detection")


def test_resolve_accepts_a_scalar_excluded_value_including_zero():
    resolved = resolve_classes(
        {"classes": {"cell": [1]}, "drop": 0}, excluded_key="drop", subject="detection"
    )

    assert resolved == (1, ("cell",), {1: 0}, (0,))
