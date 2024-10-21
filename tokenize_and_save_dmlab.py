import os
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from tokenizer.utils_vq import vq_get_encoder_decoder

def tokenize_and_save_dataset(cfg, split='training'):
    # Initialize the tokenizer
    encode_fn, _ = vq_get_encoder_decoder(cfg, device='cpu')  # Use 'cpu' or 'cuda' as appropriate

    # Paths
    data_dir = Path(cfg.save_dir) / split
    tokenized_dir = Path(cfg.save_dir) / f"{split}_tokenized"
    tokenized_dir.mkdir(parents=True, exist_ok=True)

    # Get list of .npz files
    npz_files = sorted(list(data_dir.glob("**/*.npz")), key=lambda x: x.name)

    for npz_file in tqdm(npz_files, desc=f"Tokenizing {split} data"):
        # Load the data
        data = np.load(npz_file)

        video = data['video']  # Shape: (total_frames, H, W, 3)
        actions = data['actions']  # Shape: (total_frames, )

        # Convert video to tensor and rearrange dimensions
        video_tensor = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2)  # Shape: (T, 3, H, W)

        # Tokenize the video frames
        with torch.no_grad():
            # If the video is too long, you might need to process it in batches to avoid memory issues
            batch_size = 64  # Adjust based on your memory constraints
            tokens_list = []
            for i in range(0, len(video_tensor), batch_size):
                batch = video_tensor[i:i+batch_size]
                tokens = encode_fn(batch)  # Shape: (batch_size, h, w)
                tokens_list.append(tokens)

            video_tokens = torch.cat(tokens_list, dim=0)  # Shape: (total_frames, h, w)

        # Save the tokenized data
        tokenized_file = tokenized_dir / npz_file.name
        np.savez_compressed(
            tokenized_file,
            video_tokens=video_tokens.numpy(),
            actions=actions,
        )

    print(f"Tokenization complete. Tokenized data saved in {tokenized_dir}")

if __name__ == "__main__":
    from omegaconf import DictConfig

    cfg = DictConfig({
        'save_dir': 'data/dmlab/',
        'tokenizer': {
            'name': "sd_vq_f8",
            'vocab_size': 16385,
            'token_len': 1024,
            'latent_size': 8,
            'ckpt_path': "./pretrained_ckpt/ldm/vq-f8.ckpt",
            'config_path': "./configurations/tokenizers/sd_vq_f8.yaml",
            'mask_token_id': 16384,
            'mask_token_reindex': 0,
        },
        'input_tensor_type': 'bt', 
    })

    tokenize_and_save_dataset(cfg, split='training')
    tokenize_and_save_dataset(cfg, split='validation')
