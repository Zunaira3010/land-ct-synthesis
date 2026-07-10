"""
Lung segmentation via the `lungmask` package (Hofmanninger et al. U-Net, ref [13] in the paper
-- "Lung regions in both datasets were segmented with a pre-trained open-source U-Net [13]").

We use lungmask directly rather than retraining anything -- this is a faithful match to the
paper's cited method, not an approximation (see docs/02_dataset_pipeline.md).
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def segment_lungs(image: sitk.Image, model_name: str = "R231") -> np.ndarray:
    """Run lungmask on a native-resolution CT SimpleITK image (HU values, un-resampled).

    Run this BEFORE resample_to_spacing(), same as nodule mask generation -- see
    docs/03_preprocessing.md step 6, which resamples CT+masks together afterward so
    everything stays aligned.

    Returns a boolean array, shape == sitk.GetArrayFromImage(image).shape, True inside the
    lungs. lungmask's default "R231" model returns a label map (0=background, 1=right lung,
    2=left lung); we collapse both lung labels into one binary mask since LAND's mask encoding
    doesn't distinguish left/right (paper: lungs get a single constant value of 0.5).
    """
    from lungmask import LMInferer

    inferer = LMInferer(modelname=model_name)
    label_map = inferer.apply(image)  # numpy array, same shape/order as sitk.GetArrayFromImage
    return label_map > 0
