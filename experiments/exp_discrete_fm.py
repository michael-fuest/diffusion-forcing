from datasets.video import (
    MinecraftVideoDataset,
    DmlabVideoDataset,
)
from algorithms.diffusion_forcing.df_discrete_fm import DiscreteFMVideo
from .exp_base import BaseLightningExperiment


class DiscreteFMVideoPredictionExperiment(BaseLightningExperiment):
    """
    A video prediction experiment that uses discrete flow matching.
    """

    compatible_algorithms = dict(
        df_discrete_fm=DiscreteFMVideo,
    )

    compatible_datasets = dict(
        # video datasets
        video_minecraft=MinecraftVideoDataset,
        video_dmlab=DmlabVideoDataset,
    )
