from typing import Sequence
import tarfile
import io
import torch
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from internetarchive import download
from .base_video_dataset import BaseVideoDataset
from tokenizer.utils_vq import vq_get_encoder_decoder


class DmlabVideoDataset(BaseVideoDataset):
    """
    Dmlab dataset
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        if split == "test":
            split = "validation"
        super().__init__(cfg, split)
        self.data_paths = self.get_data_paths(split)
        print(f"Number of data paths: {len(self.data_paths)}")


    def download_dataset(self) -> Sequence[int]:

        part_suffixes = ["aa", "ab", "ac"]
        for part_suffix in part_suffixes:
            identifier = f"dmlab_dataset_{part_suffix}"
            file_name = f"dmlab.tar.part{part_suffix}"
            download(identifier, file_name, destdir=self.save_dir, verbose=True)

        combined_bytes = io.BytesIO()
        for part_suffix in part_suffixes:
            identifier = f"dmlab_dataset_{part_suffix}"
            file_name = f"dmlab.tar.part{part_suffix}"
            part_file = self.save_dir / identifier / file_name
            with open(part_file, "rb") as part:
                combined_bytes.write(part.read())
        combined_bytes.seek(0)
        with tarfile.open(fileobj=combined_bytes, mode="r") as combined_archive:
            combined_archive.extractall(self.save_dir)
        (self.save_dir / "dmlab/test").rename(self.save_dir / "validation")
        (self.save_dir / "dmlab/train").rename(self.save_dir / "training")
        (self.save_dir / "dmlab").rmdir()
        for part_suffix in part_suffixes:
            identifier = f"dmlab_dataset_{part_suffix}"
            file_name = f"dmlab.tar.part{part_suffix}"
            part_file = self.save_dir / identifier / file_name
            part_file.rmdir()

    def get_data_paths(self, split):
        if self.cfg.tokenize:
            data_dir = self.save_dir / f"{split}_tokenized"
        else:
            data_dir = self.save_dir / split
        paths = sorted(list(data_dir.glob("**/*.npz")), key=lambda x: x.name)
        return paths


    def get_data_lengths(self, split):
        lengths = [300] * len(self.get_data_paths(split))
        return lengths

    def __getitem__(self, idx):
        idx = self.idx_remap[idx]
        file_idx, frame_idx = self.split_idx(idx)
        data_path = self.data_paths[file_idx]
        data = np.load(data_path)
        if self.cfg.tokenize:
            video_tokens = data["video_tokens"]  # (T, H', W')
            actions = data["actions"]  # (T,)

            video_tokens = video_tokens[frame_idx : frame_idx + self.n_frames]
            actions = actions[frame_idx : frame_idx + self.n_frames]

            pad_len = self.n_frames - len(video_tokens)
            nonterminal = np.ones(self.n_frames)
            if pad_len > 0:
                video_tokens = np.pad(video_tokens, ((0, pad_len), (0, 0), (0, 0)), mode='constant')
                actions = np.pad(actions, ((0, pad_len),), mode='constant')
                nonterminal[-pad_len:] = 0

            video_tokens = torch.from_numpy(video_tokens).long()
            actions = torch.from_numpy(actions).long()
            nonterminal = torch.from_numpy(nonterminal).float()

            return (
                video_tokens[:: self.frame_skip],
                actions[:: self.frame_skip],
                nonterminal[:: self.frame_skip],
            )
        else:
            video = data["video"][frame_idx : frame_idx + self.n_frames]  # (t, h, w, 3)
            actions = data["actions"][frame_idx : frame_idx + self.n_frames]  # (t, )
            actions = np.eye(3)[actions]  # (t, 3)
            pad_len = self.n_frames - len(video)
            nonterminal = np.ones(self.n_frames)
            if pad_len > 0:
                video = np.pad(video, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='constant')
                actions = np.pad(actions, ((0, pad_len),), mode='constant')
                nonterminal[-pad_len:] = 0
            video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2).contiguous()
            video = self.transform(video)
            return (
                video[:: self.frame_skip],
                actions[:: self.frame_skip],
                nonterminal[:: self.frame_skip],
            )


if __name__ == "__main__":
    import torch
    from unittest.mock import MagicMock
    import tqdm

    cfg = MagicMock()
    cfg.resolution = 64
    cfg.external_cond_dim = 0
    cfg.n_frames = 64
    cfg.save_dir = "data/dmlab"
    cfg.validation_multiplier = 1

    dataset = DmlabVideoDataset(cfg, "training")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=16)

    for batch in tqdm.tqdm(dataloader):
        pass
