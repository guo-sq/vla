"""Frame attributes pipeline for valid_mask, sample_weight, segment_id computation."""

from openpi.training.frame_attributes_preprocessors.base import EXTRA_EPISODE_PROMPT_MAP
from openpi.training.frame_attributes_preprocessors.base import EXTRA_SKIP_STATIC_WEIGHT
from openpi.training.frame_attributes_preprocessors.base import DatasetContext
from openpi.training.frame_attributes_preprocessors.base import EpisodeBoundary
from openpi.training.frame_attributes_preprocessors.base import FrameAttributeProcessor
from openpi.training.frame_attributes_preprocessors.base import FrameAttributes
from openpi.training.frame_attributes_preprocessors.base import run_frame_attr_preprocessor_pipeline
from openpi.training.frame_attributes_preprocessors.base import run_pipeline_single_episode
from openpi.training.frame_attributes_preprocessors.indicator_preprocessor import IndicatorPreprocessor
from openpi.training.frame_attributes_preprocessors.optimality_preprocessor import OptimalityProcessor
from openpi.training.frame_attributes_preprocessors.sample_weight_preprocessor import (
    DifficultyLabelSampleWeightPreprocessor,
)
from openpi.training.frame_attributes_preprocessors.sample_weight_preprocessor import FrameWeightByDimThresholdProcessor
from openpi.training.frame_attributes_preprocessors.sample_weight_preprocessor import (
    GripperCountSampleWeightPreprocessor,
)
from openpi.training.frame_attributes_preprocessors.sample_weight_preprocessor import GripperCountSampleWeightRule
from openpi.training.frame_attributes_preprocessors.sample_weight_preprocessor import (
    RepoNameMatchSampleWeightPreprocessor,
)
from openpi.training.frame_attributes_preprocessors.sample_weight_preprocessor import (
    StaticRatioSampleWeightPreprocessor,
)
from openpi.training.frame_attributes_preprocessors.stale_head_frames_valid_mask_preprocessor import (
    StaleHeadFramesValidMaskPreprocessor,
)
from openpi.training.frame_attributes_preprocessors.static_detector import VelocityBasedStaticDetector
from openpi.training.frame_attributes_preprocessors.temporal_weight_preprocessor import TemporalWeightProcessor
from openpi.training.frame_attributes_preprocessors.valid_mask_preprocessor import GripperCountRule
from openpi.training.frame_attributes_preprocessors.valid_mask_preprocessor import GripperCountValidMaskPreprocessor
from openpi.training.frame_attributes_preprocessors.valid_mask_preprocessor import HfColumnIsValidPreprocessor
from openpi.training.frame_attributes_preprocessors.valid_mask_preprocessor import (
    PruneHeadTailStaticValidMaskPreprocessor,
)
from openpi.training.frame_attributes_preprocessors.valid_mask_preprocessor import ValidMaskGroupParams
from openpi.training.frame_attributes_preprocessors.value_prediction_preprocessor import ValuePredictionPreprocessor
from openpi.training.frame_attributes_preprocessors.value_returns_preprocessor import ValueReturnsPreprocessor

__all__ = [
    "EXTRA_EPISODE_PROMPT_MAP",
    "EXTRA_SKIP_STATIC_WEIGHT",
    "DatasetContext",
    "DifficultyLabelSampleWeightPreprocessor",
    "EpisodeBoundary",
    "FrameAttributeProcessor",
    "FrameAttributes",
    "FrameWeightByDimThresholdProcessor",
    "GripperCountRule",
    "GripperCountSampleWeightPreprocessor",
    "GripperCountSampleWeightRule",
    "GripperCountValidMaskPreprocessor",
    "HfColumnIsValidPreprocessor",
    "IndicatorPreprocessor",
    "OptimalityProcessor",
    "PruneHeadTailStaticValidMaskPreprocessor",
    "RepoNameMatchSampleWeightPreprocessor",
    "StaleHeadFramesValidMaskPreprocessor",
    "StaticRatioSampleWeightPreprocessor",
    "TemporalWeightProcessor",
    "ValidMaskGroupParams",
    "ValuePredictionPreprocessor",
    "ValueReturnsPreprocessor",
    "VelocityBasedStaticDetector",
    "run_frame_attr_preprocessor_pipeline",
    "run_pipeline_single_episode",
]
