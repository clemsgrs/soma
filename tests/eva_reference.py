"""Verbatim copies of the kaiko-ai/eva code the EVA slide-level oracles test against.

Copied from kaiko-ai/eva @ f5d80152 (``src/eva/...``; each block names its source file).
EVA is not importable in soma's test environment, so its code lives here unchanged; only
the module plumbing differs: the ``backends`` / ``samplers`` / ``base`` / ``_utils``
module references resolve to the namespaces below, and the dataset classes are reduced
to the methods that choose slides and splits (hosted by small stand-in classes whose
other state the tests provide). Do not edit the copied bodies: the oracle tests are only
as good as their fidelity to EVA.
"""

# ruff: noqa
from __future__ import annotations

import abc
import dataclasses
import functools
import glob
import os
from types import SimpleNamespace
from typing import Any, Dict, Generator, Iterable, List, Literal, Sequence, Tuple, Type

import cv2
import numpy as np
import openslide
import pandas as pd
import torch
import torch.nn as nn
from typing_extensions import override

# --- vision/data/wsi/backends/base.py: Wsi (verbatim) ---
class Wsi(abc.ABC):
    """Base class for loading data from Whole Slide Image (WSI) files."""

    def __init__(self, file_path: str, overwrite_mpp: float | None = None):
        """Initializes a Wsi object.

        Args:
            file_path: The path to the WSI file.
            overwrite_mpp: The microns per pixel (mpp) value to use when missing in WSI metadata.
        """
        self._wsi = self.open_file(file_path)
        self._overwrite_mpp = overwrite_mpp

    @abc.abstractmethod
    def open_file(self, file_path: str) -> Any:
        """Opens the WSI file.

        Args:
            file_path: The path to the WSI file.
        """

    @property
    @abc.abstractmethod
    def level_dimensions(self) -> Sequence[Tuple[int, int]]:
        """A list of (width, height) tuples for each level, from highest to lowest resolution."""

    @property
    @abc.abstractmethod
    def level_downsamples(self) -> Sequence[float]:
        """A list of downsampling factors for each level, relative to the highest resolution."""

    @property
    @abc.abstractmethod
    def mpp(self) -> float:
        """Microns per pixel at the highest resolution (level 0)."""

    @abc.abstractmethod
    def _read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        """Abstract method to read a region at a specified zoom level."""

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        """Reads and returns image data for a specified region and zoom level.

        Args:
            location: Top-left corner (x, y) to start reading at level 0.
            level: WSI level to read from.
            size: Region size as (width, height) in pixels at the selected read level.
                Remember to scale the size correctly.
        """
        self._verify_location(location, size)
        data = self._read_region(location, level, size)
        return self._read_postprocess(data)

    def get_closest_level(self, target_mpp: float) -> int:
        """Calculate the slide level that is closest to the target mpp.

        Args:
            slide: The whole-slide image object.
            target_mpp: The target microns per pixel (mpp) value.
        """
        # Calculate the mpp for each level
        level_mpps = self.mpp * np.array(self.level_downsamples)

        # Ignore levels with higher mpp
        level_mpps_filtered = level_mpps.copy()
        level_mpps_filtered[level_mpps_filtered > target_mpp] = 0

        if level_mpps_filtered.max() == 0:
            # When all levels have higher mpp than target_mpp return the level with lowest mpp
            level_idx = np.argmin(level_mpps)
        else:
            level_idx = np.argmax(level_mpps_filtered)

        return int(level_idx)

    def _verify_location(self, location: Tuple[int, int], size: Tuple[int, int]) -> None:
        """Verifies that the requested region is within the slide dimensions.

        Args:
            location: Top-left corner (x, y) to start reading at level 0.
            size: Region size as (width, height) in pixels at the selected read level.
        """
        x_max, y_max = self.level_dimensions[0]
        x_scale = x_max / self.level_dimensions[0][0]
        y_scale = y_max / self.level_dimensions[0][1]

        if (
            int(location[0] + x_scale * size[0]) > x_max
            or int(location[1] + y_scale * size[1]) > y_max
        ):
            raise ValueError(f"Out of bounds region: {location}, {size}")

    def _read_postprocess(self, data: np.ndarray) -> np.ndarray:
        """Post-processes the read region data.

        Args:
            data: The read region data as a numpy array of shape (height, width, channels).
        """
        # Change color to white where the alpha channel is 0
        if data.shape[2] == 4:
            data[data[:, :, 3] == 0] = 255

        return data[:, :, :3]


base = SimpleNamespace(Wsi=Wsi)


# --- vision/data/wsi/backends/openslide.py: WsiOpenslide (verbatim) ---
class WsiOpenslide(base.Wsi):
    """Class for loading data from WSI files using the OpenSlide library."""

    _wsi: openslide.OpenSlide

    @override
    def open_file(self, file_path: str) -> openslide.OpenSlide:
        return openslide.OpenSlide(file_path)

    @property
    @override
    def level_dimensions(self) -> Sequence[Tuple[int, int]]:
        return self._wsi.level_dimensions

    @property
    @override
    def level_downsamples(self) -> Sequence[float]:
        return self._wsi.level_downsamples

    @property
    @override
    def mpp(self) -> float:
        # TODO: add overwrite_mpp class attribute to allow setting a default value
        if self._wsi.properties.get(openslide.PROPERTY_NAME_MPP_X) and self._wsi.properties.get(
            openslide.PROPERTY_NAME_MPP_Y
        ):
            x_mpp = float(self._wsi.properties[openslide.PROPERTY_NAME_MPP_X])
            y_mpp = float(self._wsi.properties[openslide.PROPERTY_NAME_MPP_Y])
        elif (
            self._wsi.properties.get("tiff.XResolution")
            and self._wsi.properties.get("tiff.YResolution")
            and self._wsi.properties.get("tiff.ResolutionUnit")
        ):
            unit = self._wsi.properties.get("tiff.ResolutionUnit")
            if unit not in _conversion_factor_to_micrometer:
                raise ValueError(f"Unit {unit} not supported.")

            conversion_factor = float(_conversion_factor_to_micrometer.get(unit))  # type: ignore
            x_mpp = conversion_factor / float(self._wsi.properties["tiff.XResolution"])
            y_mpp = conversion_factor / float(self._wsi.properties["tiff.YResolution"])
        else:
            raise ValueError("`mpp` cannot be obtained for this slide.")

        return (x_mpp + y_mpp) / 2.0

    @override
    def _read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        return np.array(self._wsi.read_region(location, level, size))


# --- vision/data/wsi/backends/openslide.py: _conversion_factor_to_micrometer (verbatim) ---
_conversion_factor_to_micrometer = {
    "meter": 10**6,
    "decimeter": 10**5,
    "centimeter": 10**4,
    "millimeter": 10**3,
    "micrometer": 1,
    "nanometer": 10**-3,
    "picometer": 10**-6,
    "femtometer": 10**-9,
}


# --- vision/data/wsi/patching/mask.py: Mask (verbatim) ---
@dataclasses.dataclass
class Mask:
    """A class to store the mask of a whole-slide image."""

    mask_array: np.ndarray
    """Binary mask array where 1s represent the foreground and 0s represent the background."""

    mask_level_idx: int
    """WSI level index at which the mask_array was extracted."""

    scale_factors: Tuple[float, float]
    """Factors to scale x/y coordinates from mask_level_idx to level 0."""


# --- vision/data/wsi/patching/mask.py: get_mask (verbatim) ---
def get_mask(
    wsi: Wsi,
    mask_level_idx: int,
    saturation_threshold: int = 20,
    median_blur_kernel_size: int | None = None,
    fill_holes: bool = False,
    holes_kernel_size: Tuple[int, int] = (7, 7),
    use_otsu: bool = False,
) -> Mask:
    """Generates a binary foreground mask for a given WSI.

    The is a simplified version of the algorithm proposed in [1] (CLAM):
    1. Convert the image to the HSV color space (easier to seperate specific colors with RGB).
    2. (optional) Apply a median blur to the saturation channel to reduce noise
        & closing small gaps in the mask. While this yields cleaner masks, this step is the most
        computationally expensive and thus disabled by default (CLAM uses a value of 7).
    3. Calculate binary mask by thresholding accross the saturation channel.

    [1] Lu, Ming Y., et al. "Data-efficient and weakly supervised computational
        pathology on whole-slide images." Nature biomedical engineering 5.6 (2021): 555-570.
        https://github.com/mahmoodlab/CLAM

    Args:
        wsi: The WSI object.
        mask_level_idx: The level index of the WSI at which we want to extract the mask.
        saturation_threshold: The threshold value for the saturation channel.
        median_blur_kernel_size: Kernel size for the median blur operation.
        holes_kernel_size: The size of the kernel for morphological operations to fill holes.
        fill_holes: Whether to fill holes in the mask.
        use_otsu: Whether to use Otsu's method for the thresholding operation. If False,
            a fixed threshold value is used.

    Returns: A Mask object instance.
    """
    image = wsi.read_region((0, 0), mask_level_idx, wsi.level_dimensions[mask_level_idx])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image = (
        cv2.medianBlur(image[:, :, 1], median_blur_kernel_size)
        if median_blur_kernel_size
        else image[:, :, 1]
    )

    threshold_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU if use_otsu else cv2.THRESH_BINARY
    _, mask_array = cv2.threshold(image, saturation_threshold, 1, threshold_type)

    if fill_holes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, holes_kernel_size)
        mask_array = cv2.dilate(mask_array, kernel, iterations=1)
        contour, _ = cv2.findContours(mask_array, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(mask_array, [cnt], 0, (1,), -1)

    mask_array = mask_array.astype(np.uint8)
    scale_factors = (
        wsi.level_dimensions[0][0] / wsi.level_dimensions[mask_level_idx][0],
        wsi.level_dimensions[0][1] / wsi.level_dimensions[mask_level_idx][1],
    )

    return Mask(mask_array=mask_array, mask_level_idx=mask_level_idx, scale_factors=scale_factors)


# --- vision/data/wsi/patching/mask.py: get_mask_level (verbatim) ---
def get_mask_level(
    wsi: Wsi,
    width: int,
    height: int,
    target_mpp: float,
    min_mask_patch_pixels: int = 3 * 3,
) -> int:
    """For performance reasons, we generate the mask at the lowest resolution level possible.

    However, if minimum resolution level has too few pixels, the patches scaled to that level will
    be too small or even collapse to a single pixel. This function allows to find the lowest
    resolution level that yields mask patches with at least `min_mask_patch_pixels` pixels.

    Args:
        wsi: The WSI object.
        width: The width of the patches to be extracted, in pixels (at target_mpp).
        height: The height of the patches to be extracted, in pixels.
        target_mpp: The target microns per pixel (mpp) for the patches.
        min_mask_patch_pixels: The minimum number of pixels required for the mask patches.
            Mask patch refers to width / height at target_mpp scaled down to the WSI level
            at which the mask is generated.
    """
    level_mpps = wsi.mpp * np.array(wsi.level_downsamples)
    mask_level_idx = None

    for level_idx, level_mpp in reversed(list(enumerate(level_mpps))):
        mpp_ratio = target_mpp / level_mpp
        scaled_width, scaled_height = int(mpp_ratio * width), int(mpp_ratio * height)

        if scaled_width * scaled_height >= min_mask_patch_pixels:
            mask_level_idx = level_idx
            break

    if mask_level_idx is None:
        raise ValueError("No level with the specified minimum number of patch pixels available.")

    return mask_level_idx


# --- vision/data/wsi/patching/samplers/_utils.py: get_grid_coords_and_indices (verbatim) ---
def get_grid_coords_and_indices(
    layer_shape: Tuple[int, int],
    width: int,
    height: int,
    overlap: Tuple[int, int],
    shuffle: bool = True,
    seed: int = 42,
):
    """Get grid coordinates and indices.

    Args:
        layer_shape: The shape of the layer.
        width: The width of the patches.
        height: The height of the patches.
        overlap: The overlap between patches in the grid.
        shuffle: Whether to shuffle the indices.
        seed: The random seed.
    """
    x_range = range(0, layer_shape[0] - width + 1, width - overlap[0])
    y_range = range(0, layer_shape[1] - height + 1, height - overlap[1])
    x_y = [(x, y) for x in x_range for y in y_range]

    indices = list(range(len(x_y)))
    if shuffle:
        random_generator = np.random.default_rng(seed)
        random_generator.shuffle(indices)
    return x_y, indices


# --- vision/data/wsi/patching/samplers/_utils.py: validate_dimensions (verbatim) ---
def validate_dimensions(width: int, height: int, layer_shape: Tuple[int, int]) -> None:
    """Checks if the width / height is bigger than the layer shape.

    Args:
        width: The width of the patches.
        height: The height of the patches.
        layer_shape: The shape of the layer.
    """
    if width > layer_shape[0] or height > layer_shape[1]:
        raise ValueError("The width / height cannot be bigger than the layer shape.")


_utils = SimpleNamespace(
    get_grid_coords_and_indices=get_grid_coords_and_indices,
    validate_dimensions=validate_dimensions,
)


# --- vision/data/wsi/patching/samplers/base.py: Sampler (verbatim) ---
class Sampler(abc.ABC):
    """Base class for samplers."""

    @abc.abstractmethod
    def sample(
        self,
        width: int,
        height: int,
        layer_shape: Tuple[int, int],
        mask: Mask | None = None,
    ) -> Generator[Tuple[int, int], None, None]:
        """Sample patche coordinates.

        Args:
            width: The width of the patches.
            height: The height of the patches.
            layer_shape: The shape of the layer.
            mask: Tuple containing the mask array and the scaling factor with respect to the
                provided layer_shape. Optional, only required for samplers with foreground
                filtering.

        Returns:
            A generator producing sampled patch coordinates.
        """


# --- vision/data/wsi/patching/samplers/base.py: ForegroundSampler (verbatim) ---
class ForegroundSampler(Sampler):
    """Base class for samplers with foreground filtering capabilities."""

    @abc.abstractmethod
    def is_foreground(
        self,
        mask: Mask,
        x: int,
        y: int,
        width: int,
        height: int,
        min_foreground_ratio: float,
    ) -> bool:
        """Check if a patch contains sufficient foreground."""


base.Sampler = Sampler
base.ForegroundSampler = ForegroundSampler


# --- vision/data/wsi/patching/samplers/foreground_grid.py: ForegroundGridSampler (verbatim) ---
class ForegroundGridSampler(base.ForegroundSampler):
    """Sample patches based on a grid, only returning patches containing foreground."""

    def __init__(
        self,
        max_samples: int = 20,
        overlap: Tuple[int, int] = (0, 0),
        min_foreground_ratio: float = 0.35,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        """Initializes the sampler.

        Args:
            max_samples: The maximum number of samples to return.
            overlap: The overlap between patches in the grid.
            min_foreground_ratio: The minimum amount of foreground
                within a sampled patch.
            shuffle: Whether to shuffle the grid indices before sampling.
                If True, patches are randomly distributed across the WSI.
                If False, patches are sampled in column-major order from top-left.
            seed: The random seed.
        """
        self.max_samples = max_samples
        self.overlap = overlap
        self.min_foreground_ratio = min_foreground_ratio
        self.shuffle = shuffle
        self.seed = seed

    def sample(
        self,
        width: int,
        height: int,
        layer_shape: Tuple[int, int],
        mask: Mask,
    ):
        """Sample patches from a grid containing foreground.

        Args:
            width: The width of the patches.
            height: The height of the patches.
            layer_shape: The shape of the layer.
            mask: The mask of the image.
        """
        _utils.validate_dimensions(width, height, layer_shape)
        x_y, indices = _utils.get_grid_coords_and_indices(
            layer_shape, width, height, self.overlap, shuffle=self.shuffle, seed=self.seed
        )

        count = 0
        for i in indices:
            if count >= self.max_samples:
                break

            if self.is_foreground(
                mask=mask,
                x=x_y[i][0],
                y=x_y[i][1],
                width=width,
                height=height,
                min_foreground_ratio=self.min_foreground_ratio,
            ):
                count += 1
                yield x_y[i]

    def is_foreground(
        self,
        mask: Mask,
        x: int,
        y: int,
        width: int,
        height: int,
        min_foreground_ratio: float,
    ) -> bool:
        """Check if a patch contains sufficient foreground.

        Args:
            mask: The mask of the image.
            x: The x-coordinate of the patch.
            y: The y-coordinate of the patch.
            width: The width of the patch.
            height: The height of the patch.
            min_foreground_ratio: The minimum amount of foreground in the patch.
        """
        x_, y_ = self._scale_coords(x, y, mask.scale_factors)
        width_, height_ = self._scale_coords(width, height, mask.scale_factors)
        patch_mask = mask.mask_array[y_ : y_ + height_, x_ : x_ + width_]
        return patch_mask.sum() / patch_mask.size >= min_foreground_ratio

    def _scale_coords(
        self,
        x: int,
        y: int,
        scale_factors: Tuple[float, float],
    ) -> Tuple[int, int]:
        return int(x / scale_factors[0]), int(y / scale_factors[1])


samplers = SimpleNamespace(
    Sampler=Sampler,
    ForegroundSampler=ForegroundSampler,
    ForegroundGridSampler=ForegroundGridSampler,
)
backends = SimpleNamespace(wsi_backend=lambda backend="openslide": WsiOpenslide)


# --- vision/data/wsi/patching/coordinates.py: PatchCoordinates (verbatim) ---
@dataclasses.dataclass
class PatchCoordinates:
    """A class to store coordinates of patches from a whole-slide image.

    Args:
        x_y: A list of (x, y) coordinates of the patches (refer to level 0).
        width: The width of the patches, in pixels (refers to level_idx).
        height: The height of the patches, in pixels (refers to level_idx).
        level_idx: The level index at which to extract the patches.
        mask: The foreground mask of the wsi.
    """

    x_y: List[Tuple[int, int]]
    width: int
    height: int
    level_idx: int
    mask: Mask | None = None

    @classmethod
    def from_file(
        cls,
        wsi_path: str,
        width: int,
        height: int,
        sampler: samplers.Sampler,
        target_mpp: float,
        overwrite_mpp: float | None = None,
        backend: str = "openslide",
    ) -> "PatchCoordinates":
        """Create a new instance of PatchCoordinates from a whole-slide image file.

        Patches will be read from the level that is closest to the specified target_mpp.

        Args:
            wsi_path: The path to the whole-slide image file.
            width: The width of the patches to be extracted, in pixels.
            height: The height of the patches to be extracted, in pixels.
            target_mpp: The target microns per pixel (mpp) for the patches.
            overwrite_mpp: The microns per pixel (mpp) value to use when missing in WSI metadata.
            sampler: The sampler to use for sampling patch coordinates.
            backend: The backend to use for reading the whole-slide images.
        """
        wsi = backends.wsi_backend(backend)(wsi_path, overwrite_mpp)

        # Sample patch coordinates at level 0
        mpp_ratio_0 = target_mpp / wsi.mpp
        sample_args = {
            "width": int(mpp_ratio_0 * width),
            "height": int(mpp_ratio_0 * height),
            "layer_shape": wsi.level_dimensions[0],
        }
        if isinstance(sampler, samplers.ForegroundSampler):
            mask_level_idx = get_mask_level(wsi, width, height, target_mpp)
            sample_args["mask"] = get_mask(wsi, mask_level_idx)

        x_y = list(sampler.sample(**sample_args))

        # Scale dimensions to level that is closest to the target_mpp
        level_idx = wsi.get_closest_level(target_mpp)
        mpp_ratio = target_mpp / (wsi.mpp * wsi.level_downsamples[level_idx])
        scaled_width, scaled_height = int(mpp_ratio * width), int(mpp_ratio * height)

        return cls(x_y, scaled_width, scaled_height, level_idx, sample_args.get("mask"))

    def to_dict(self, include_keys: List[str] | None = None) -> Dict[str, Any]:
        """Convert the coordinates to a dictionary."""
        include_keys = include_keys or ["x_y", "width", "height", "level_idx"]
        coord_dict = dataclasses.asdict(self)
        if include_keys:
            coord_dict = {key: coord_dict[key] for key in include_keys}
        return coord_dict


# --- core/data/splitting/stratified.py: stratified_split (verbatim) ---
def stratified_split(
    samples: Sequence[Any] | Iterable[Any],
    targets: Sequence[Any] | Iterable[Any],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float = 0.0,
    groups: Sequence[Any] | Iterable[Any] | None = None,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int] | None]:
    """Splits the samples into stratified train, validation, and test (optional) sets.

    Args:
        samples: The samples to split.
        targets: The corresponding targets used for stratification.
        train_ratio: The ratio of the training set.
        val_ratio: The ratio of the validation set.
        test_ratio: The ratio of the test set (optional).
        groups: Optional group labels for group-wise stratification.
        seed: The seed for reproducibility.

    Returns:
        The indices of the train, validation, and test sets.
    """
    samples_seq = samples if isinstance(samples, (list, tuple)) else list(samples)
    targets_seq = targets if isinstance(targets, (list, tuple)) else list(targets)

    if train_ratio + val_ratio + test_ratio > 1.0:
        raise ValueError("The sum of the ratios must be lower or equal to 1")

    if len(samples_seq) != len(targets_seq):
        raise ValueError("The number of samples and targets must be equal.")

    if groups is not None:
        groups_seq = groups if isinstance(groups, (list, tuple)) else list(groups)
        if len(groups_seq) != len(samples_seq):
            raise ValueError("The number of samples and groups must be equal.")

        unique_groups, group_indices = np.unique(groups_seq, return_inverse=True)
        group_targets = np.array(
            [targets_seq[np.where(group_indices == i)[0][0]] for i in range(len(unique_groups))]
        )

        train_g, val_g, test_g = stratified_split(
            samples=unique_groups.tolist(),
            targets=group_targets.tolist(),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

        def map_indices(g_list):
            if g_list is None:
                return []
            selected_groups = unique_groups[g_list]
            return np.where(np.isin(groups_seq, selected_groups))[0].tolist()

        return map_indices(train_g), map_indices(val_g), map_indices(test_g) or None

    use_all_samples = train_ratio + val_ratio + test_ratio == 1
    random_generator = np.random.default_rng(seed)
    unique_classes, y_indices = np.unique(targets_seq, return_inverse=True)
    n_classes = unique_classes.shape[0]

    train_indices, val_indices, test_indices = [], [], []

    for c in range(n_classes):
        class_indices = np.where(y_indices == c)[0]
        random_generator.shuffle(class_indices)

        n_train = int(np.floor(train_ratio * len(class_indices))) or 1
        n_val = (
            len(class_indices) - n_train
            if test_ratio == 0.0 and use_all_samples
            else int(np.floor(val_ratio * len(class_indices))) or 1
        )

        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train : n_train + n_val])
        if test_ratio > 0.0:
            n_test = (
                len(class_indices) - n_train - n_val
                if use_all_samples
                else int(np.floor(test_ratio * len(class_indices))) or 1
            )
            test_indices.extend(class_indices[n_train + n_val : n_train + n_val + n_test])

    return train_indices, val_indices, test_indices or None


splitting = SimpleNamespace(stratified_split=stratified_split)


class Camelyon16Files:
    """Stand-in host for ``Camelyon16``'s slide-listing methods (set ``_root``)."""

    # --- vision/data/datasets/classification/camelyon16.py: Camelyon16 (verbatim members) ---
    _val_slides = [
        "normal_010",
        "normal_013",
        "normal_016",
        "normal_017",
        "normal_019",
        "normal_020",
        "normal_025",
        "normal_030",
        "normal_031",
        "normal_032",
        "normal_052",
        "normal_056",
        "normal_057",
        "normal_067",
        "normal_076",
        "normal_079",
        "normal_085",
        "normal_095",
        "normal_098",
        "normal_099",
        "normal_101",
        "normal_102",
        "normal_105",
        "normal_106",
        "normal_109",
        "normal_129",
        "normal_132",
        "normal_137",
        "normal_142",
        "normal_143",
        "normal_148",
        "normal_152",
        "tumor_001",
        "tumor_005",
        "tumor_011",
        "tumor_012",
        "tumor_013",
        "tumor_019",
        "tumor_031",
        "tumor_037",
        "tumor_043",
        "tumor_046",
        "tumor_057",
        "tumor_065",
        "tumor_069",
        "tumor_071",
        "tumor_073",
        "tumor_079",
        "tumor_080",
        "tumor_081",
        "tumor_082",
        "tumor_085",
        "tumor_097",
        "tumor_109",
    ]
    """Validation slide names, same as the ones in patch camelyon."""

    @property
    def classes(self) -> List[str]:
        return ["normal", "tumor"]

    def _load_file_paths(self, split: Literal["train", "val", "test"] | None = None) -> List[str]:
        """Loads the file paths of the corresponding dataset split."""
        train_paths, val_paths = [], []
        for path in glob.glob(os.path.join(self._root, "training/**/*.tif")):
            if self._get_id_from_path(path) in self._val_slides:
                val_paths.append(path)
            else:
                train_paths.append(path)
        test_paths = glob.glob(os.path.join(self._root, "testing/images", "*.tif"))

        match split:
            case "train":
                paths = train_paths
            case "val":
                paths = val_paths
            case "test":
                paths = test_paths
            case None:
                paths = train_paths + val_paths + test_paths
            case _:
                raise ValueError("Invalid split. Use 'train', 'val' or `None`.")
        return sorted([os.path.relpath(path, self._root) for path in paths])

    def _get_id_from_path(self, file_path: str) -> str:
        """Extracts the slide ID from the file path."""
        return os.path.basename(file_path).replace(".tif", "")

    def _get_class_from_path(self, file_path: str) -> str:
        """Extracts the class name from the file path."""
        class_name = self._get_id_from_path(file_path).split("_")[0]
        if class_name not in self.classes:
            raise ValueError(f"Invalid class name '{class_name}' in file path '{file_path}'.")
        return class_name

    def __init__(self, root: str) -> None:
        self._root = root


class PANDAFiles:
    """Stand-in host for ``PANDA``'s split methods (set ``_root`` and ``annotations``)."""

    # --- vision/data/datasets/classification/panda.py: PANDA (verbatim members) ---
    _train_split_ratio: float = 0.7
    """Train split ratio."""

    _val_split_ratio: float = 0.15
    """Validation split ratio."""

    _test_split_ratio: float = 0.15
    """Test split ratio."""

    def _load_file_paths(self, split: Literal["train", "val", "test"] | None = None) -> List[str]:
        """Loads the file paths of the corresponding dataset split."""
        image_dir = os.path.join(self._root, "train_images")
        file_paths = sorted(glob.glob(os.path.join(image_dir, "*.tiff")))
        file_paths = [os.path.relpath(path, self._root) for path in file_paths]
        if len(file_paths) != len(self.annotations):
            raise ValueError(
                f"Expected {len(self.annotations)} images, found {len(file_paths)} in {image_dir}."
            )
        file_paths = self._filter_noisy_labels(file_paths)
        targets = [self._get_target_from_path(file_path) for file_path in file_paths]

        train_indices, val_indices, test_indices = splitting.stratified_split(
            samples=file_paths,
            targets=targets,
            train_ratio=self._train_split_ratio,
            val_ratio=self._val_split_ratio,
            test_ratio=self._test_split_ratio,
            seed=self._seed,
        )

        match split:
            case "train":
                return [file_paths[i] for i in train_indices]
            case "val":
                return [file_paths[i] for i in val_indices]
            case "test":
                return [file_paths[i] for i in test_indices or []]
            case None:
                return file_paths
            case _:
                raise ValueError("Invalid split. Use 'train', 'val', 'test' or `None`.")

    def _filter_noisy_labels(self, file_paths: List[str]):
        is_noisy_filter = self.annotations["noise_ratio_10"] == 0
        non_noisy_image_ids = set(self.annotations.loc[~is_noisy_filter].index)
        filtered_file_paths = [
            file_path
            for file_path in file_paths
            if self._get_id_from_path(file_path) in non_noisy_image_ids
        ]
        return filtered_file_paths

    def _get_target_from_path(self, file_path: str) -> int:
        return self.annotations.loc[self._get_id_from_path(file_path), "isup_grade"]

    def _get_id_from_path(self, file_path: str) -> str:
        return os.path.basename(file_path).replace(".tiff", "")

    def __init__(self, root: str, annotations_csv: str, seed: int = 42) -> None:
        self._root = root
        self._seed = seed
        # PANDA.annotations (verbatim body)
        self.annotations = pd.read_csv(annotations_csv, index_col="image_id")


class PANDASmallFiles(PANDAFiles):
    """Stand-in for ``PANDASmall``."""

    # --- vision/data/datasets/classification/panda.py: PANDASmall (verbatim members) ---
    _train_split_ratio: float = 0.1
    """Train split ratio."""

    _val_split_ratio: float = 0.05
    """Validation split ratio."""

    _test_split_ratio: float = 0.05
    """Test split ratio."""



# --- core/models/networks/mlp.py: MLP (verbatim) ---
class MLP(nn.Module):
    """A Multi-layer Perceptron (MLP) network."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layer_sizes: Tuple[int, ...] | None = None,
        hidden_activation_fn: Type[torch.nn.Module] | None = nn.ReLU,
        output_activation_fn: Type[torch.nn.Module] | None = None,
        dropout: float = 0.0,
    ) -> None:
        """Initializes the MLP.

        Args:
            input_size: The number of input features.
            output_size: The number of output features.
            hidden_layer_sizes: A list specifying the number of units in each hidden layer.
            dropout: Dropout probability for hidden layers.
            hidden_activation_fn: Activation function to use for hidden layers. Default is ReLU.
            output_activation_fn: Activation function to use for the output layer. Default is None.
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes if hidden_layer_sizes is not None else ()
        self.hidden_activation_fn = hidden_activation_fn
        self.output_activation_fn = output_activation_fn
        self.dropout = dropout

        self._network = self._build_network()

    def _build_network(self) -> nn.Sequential:
        """Builds the neural network's layers and returns a nn.Sequential container."""
        layers = []
        prev_size = self.input_size
        for size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            if self.hidden_activation_fn is not None:
                layers.append(self.hidden_activation_fn())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev_size = size

        layers.append(nn.Linear(prev_size, self.output_size))
        if self.output_activation_fn is not None:
            layers.append(self.output_activation_fn())

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the MLP.

        Args:
            x: The input tensor.

        Returns:
            The output of the network.
        """
        return self._network(x)


# --- vision/models/networks/abmil.py: ABMIL (verbatim) ---
class ABMIL(torch.nn.Module):
    """ABMIL network for multiple instance learning classification tasks.

    Takes an array of patch level embeddings per slide as input. This implementation supports
    batched inputs of shape (`batch_size`, `n_instances`, `input_size`). For slides with less
    than `n_instances` patches, you can apply padding and provide a mask tensor to the forward
    pass.

    The original implementation from [1] was used as a reference:
    https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py

    Notes:
        - use_bias: The paper didn't use bias in their formalism, but their published
        example code inadvertently does.
        - To prevent dot product similarities near-equal due to concentration of measure
        as a consequence of large input embedding dimensionality (>128), we added the
        option to project the input embeddings to a lower dimensionality

    [1] Maximilian Ilse, Jakub M. Tomczak, Max Welling, "Attention-based Deep Multiple
        Instance Learning", 2018
        https://arxiv.org/abs/1802.04712
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        projected_input_size: int | None,
        hidden_size_attention: int = 128,
        hidden_sizes_mlp: tuple = (128, 64),
        use_bias: bool = True,
        dropout_input_embeddings: float = 0.0,
        dropout_attention: float = 0.0,
        dropout_mlp: float = 0.0,
        pad_value: int | float | None = float("-inf"),
    ) -> None:
        """Initializes the ABMIL network.

        Args:
            input_size: input embedding dimension
            output_size: number of classes
            projected_input_size: size of the projected input. if `None`, no projection is
                performed.
            hidden_size_attention: hidden dimension in attention network
            hidden_sizes_mlp: dimensions for hidden layers in last mlp
            use_bias: whether to use bias in the attention network
            dropout_input_embeddings: dropout rate for the input embeddings
            dropout_attention: dropout rate for the attention network and classifier
            dropout_mlp: dropout rate for the final MLP network
            pad_value: Value indicating padding in the input tensor. If specified, entries with
                this value in the will be masked. If set to `None`, no masking is applied.
        """
        super().__init__()

        self._pad_value = pad_value

        if projected_input_size:
            self.projector = nn.Sequential(
                nn.Linear(input_size, projected_input_size, bias=True),
                nn.Dropout(p=dropout_input_embeddings),
            )
            input_size = projected_input_size
        else:
            self.projector = nn.Dropout(p=dropout_input_embeddings)

        self.gated_attention = GatedAttention(
            input_dim=input_size,
            hidden_dim=hidden_size_attention,
            dropout=dropout_attention,
            n_classes=1,
            use_bias=use_bias,
        )

        self.classifier = MLP(
            input_size=input_size,
            output_size=output_size,
            hidden_layer_sizes=hidden_sizes_mlp,
            dropout=dropout_mlp,
            hidden_activation_fn=nn.ReLU,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_tensor: Tensor with expected shape of (batch_size, n_instances, input_size).
        """
        input_tensor, mask = self._mask_values(input_tensor, self._pad_value)

        # (batch_size, n_instances, input_size) -> (batch_size, n_instances, projected_input_size)
        input_tensor = self.projector(input_tensor)

        attention_logits = self.gated_attention(input_tensor)  # (batch_size, n_instances, 1)
        if mask is not None:
            # fill masked values with -inf, which will yield 0s after softmax
            attention_logits = attention_logits.masked_fill(mask, float("-inf"))

        attention_weights = nn.functional.softmax(attention_logits, dim=1)
        # (batch_size, n_instances, 1)

        attention_result = torch.matmul(torch.transpose(attention_weights, 1, 2), input_tensor)
        # (batch_size, 1, hidden_size_attention)

        attention_result = torch.squeeze(attention_result, 1)  # (batch_size, hidden_size_attention)

        return self.classifier(attention_result)  # (batch_size, output_size)

    def _mask_values(self, input_tensor: torch.Tensor, pad_value: float | None):
        """Masks the padded values in the input tensor."""
        if pad_value is None:
            return input_tensor, None
        else:
            # (batch_size, n_instances, input_size)
            mask = input_tensor == pad_value

            # (batch_size, n_instances, input_size) -> (batch_size, n_instances, 1)
            mask = mask.all(dim=-1, keepdim=True)

            # Fill masked values with 0, so that they don't contribute to dense layers
            input_tensor = input_tensor.masked_fill(mask, 0)

            return input_tensor, mask


# --- vision/models/networks/abmil.py: GatedAttention (verbatim) ---
class GatedAttention(nn.Module):
    """Attention mechanism with Sigmoid Gating using 3 linear layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.25,
        n_classes: int = 1,
        use_bias: bool = True,
        activation_a: Type[nn.Module] = nn.Tanh,
        activation_b: Type[nn.Module] = nn.Sigmoid,
    ):
        """Initializes the GatedAttention network.

        Args:
            input_dim: input feature dimension
            hidden_dim: hidden layer dimension
            dropout: dropout rate
            n_classes: number of classes
            use_bias: whether to use bias in the linear layers
            activation_a: activation function for attention_a.
            activation_b: activation function for attention_b.
        """
        super().__init__()

        def make_attention(activation: nn.Module):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=use_bias), nn.Dropout(p=dropout), activation
            )

        self.attention_a = make_attention(activation_a())
        self.attention_b = make_attention(activation_b())
        self.attention_c = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        a = self.attention_a(x)  # [..., hidden_dim]
        b = self.attention_b(x)  # [..., hidden_dim]
        att = a.mul(b)  # [..., hidden_dim]
        att = self.attention_c(att)  # [..., n_classes]
        return att
