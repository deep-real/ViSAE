import os
from typing import Any, Iterator, cast, List, Optional, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import lru_cache
from hooked_model import HookedModel
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class ActivationStoreConfig:
    hook_points: List[str]
    n_batches_in_buffer: int = 128
    store_batch_size: int = 64
    train_batch_size: int = 4096
    context_size: int = 257
    d_in: int = 768  # Input dimension
    d_out: int = 768  # Output dimension for transcoder
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = torch.device('cuda:1')
    use_cached_activations: bool = False
    cached_activations_path: str = ""
    is_transcoder: bool = False
    use_patches_only: bool = False
    num_workers: int = 0

class ProbingDataset(torch.utils.data.Dataset):
    def __init__(self, source_dir="path/to/data", preprocessor=None):
        self.source_dir = source_dir
        self.image_paths = sorted([
            str(p) for p in Path(source_dir).glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        ])

        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.preprocessor:
            image = self.preprocessor(image, return_tensors="pt")["pixel_values"][0]
        return image


def collate_fn(data):
    imgs = [d[0] for d in data]
    return torch.stack(imgs, dim=0)


def collate_fn_eval(data):
    imgs = [d[0] for d in data]
    return torch.stack(imgs, dim=0), torch.tensor([d[1] for d in data])


class VisionActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs.
    """

    def __init__(self, cfg: ActivationStoreConfig, model_name="facebook/dinov2-base", create_dataloader=True, num_workers=0):
        self.cfg = cfg
        self.model = HookedModel(model_name=model_name, hook_points=cfg.hook_points, device=cfg.device, transcoder=cfg.is_transcoder)
        self.dataset = ProbingDataset(preprocessor=self.model.preprocessor)
        
        # Main dataset loader
        self.image_dataloader = DataLoader(
            self.dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=self.cfg.store_batch_size,
            # collate_fn=collate_fn,
            drop_last=True,
        )
                
        # Infinite iterator for training data
        self.image_dataloader_iter = self._batch_stream(
            self.image_dataloader, device=self.cfg.device
        )
    
        # Initialize storage buffers
        if create_dataloader:
            if self.cfg.is_transcoder:
                half_batches = self.cfg.n_batches_in_buffer // 2
                self.storage_buffer, self.storage_buffer_out = self.get_buffer(half_batches)
            else:
                self.storage_buffer = self.get_buffer(self.cfg.n_batches_in_buffer // 2)
            
            self.dataloader = self.get_data_loader()

    def __len__(self):
        return len(self.dataset) * self.cfg.context_size

    def _batch_stream(
        self, dataloader: DataLoader, device: torch.device
    ) -> Iterator[torch.Tensor]:
        """
        Infinite iterator over batches of images from a given dataloader.
        Ensures that `.requires_grad_(False)` is set and that data is moved to the specified device.
        """
        while True:
            for batch in dataloader:
                batch.requires_grad_(False)
                yield batch.to(device)

    @torch.no_grad
    def get_activations(self, batch_tokens: torch.Tensor) -> torch.Tensor:
        """Returns activations from the model, handling transcoder case if configured."""
        activations = self.model.get_activations(batch_tokens)
        if self.cfg.is_transcoder:
            activations_in = torch.stack([activations[hook_point + '_in'] for hook_point in self.model.hook_points], dim=2)
            activations_out = torch.stack([activations[hook_point + '_out'] for hook_point in self.model.hook_points], dim=2)
            return activations_in.cpu(), activations_out.cpu()
        else:
            return torch.stack([activations[hook_point] for hook_point in self.model.hook_points], dim=2).cpu()

    def get_buffer(self, n_batches_in_buffer: int) -> torch.Tensor:
        """
        Creates and returns a buffer of activations, handling transcoder case.
        """
        context_size = self.cfg.context_size
        batch_size = self.cfg.store_batch_size
        d_in = self.cfg.d_in
        total_size = batch_size * n_batches_in_buffer

        num_layers = len(self.cfg.hook_points)
        
        if self.cfg.is_transcoder:
            d_out = self.cfg.d_out
            num_out_layers = len(self.cfg.hook_points) if isinstance(self.cfg.hook_points, list) else 1

            # Initialize output buffer for transcoder
            new_buffer_out = torch.zeros(
                (total_size, context_size, num_out_layers, d_out),
                dtype=self.cfg.dtype,
                device='cpu',
            )
        
        # Generate activations buffer
        new_buffer = torch.zeros(
            (total_size, context_size, num_layers, d_in),
            dtype=self.cfg.dtype,
            device='cpu',
        )

        for start_idx in tqdm(range(0, total_size, batch_size), desc="Generating activations"):
            batch_tokens = next(self.image_dataloader_iter)
            
            if not self.cfg.is_transcoder:
                batch_activations = self.get_activations(batch_tokens)
            else:
                batch_activations_in, batch_activations_out = self.get_activations(batch_tokens)
                batch_activations = batch_activations_in

            if self.cfg.use_patches_only:
                # Remove the CLS token if we only need patches
                batch_activations = batch_activations[:, 1:, :, :]
                
            new_buffer[start_idx : start_idx + batch_size, ...] = batch_activations
            
            if self.cfg.is_transcoder:
                if self.cfg.use_patches_only:
                    batch_activations_out = batch_activations_out[:, 1:, :, :]
                new_buffer_out[start_idx : start_idx + batch_size, ...] = batch_activations_out

        # Reshape and shuffle
        new_buffer = new_buffer.reshape(-1, num_layers, d_in)
        randperm = torch.randperm(new_buffer.shape[0])
        new_buffer = new_buffer[randperm]
        
        if self.cfg.is_transcoder:
            new_buffer_out = new_buffer_out.reshape(-1, num_out_layers, d_out)
            new_buffer_out = new_buffer_out[randperm]
            return new_buffer, new_buffer_out
        
        return new_buffer

    def get_data_loader(self) -> Iterator[Any]:
        """Create a new DataLoader handling transcoder case."""
        batch_size = self.cfg.train_batch_size
        half_batches = self.cfg.n_batches_in_buffer // 2

        if self.cfg.is_transcoder:
            # Get new buffers
            new_buffer, new_buffer_out = self.get_buffer(half_batches)
            
            # Mix with storage buffers
            mixing_buffer = torch.cat([new_buffer, self.storage_buffer], dim=0)
            mixing_buffer_out = torch.cat([new_buffer_out, self.storage_buffer_out], dim=0)
            
            # Shuffle consistently
            assert mixing_buffer.shape[0] == mixing_buffer_out.shape[0]
            randperm = torch.randperm(mixing_buffer.shape[0])
            mixing_buffer = mixing_buffer[randperm]
            mixing_buffer_out = mixing_buffer_out[randperm]
            
            # Store half for next time
            self.storage_buffer = mixing_buffer[:mixing_buffer.shape[0]//2]
            self.storage_buffer_out = mixing_buffer_out[:mixing_buffer_out.shape[0]//2]
            
            # Concatenate buffers for training
            catted_buffers = torch.cat([
                mixing_buffer[mixing_buffer.shape[0]//2:],
                mixing_buffer_out[mixing_buffer.shape[0]//2:]
            ], dim=1)
            
            dataloader = iter(DataLoader(
                cast(Any, catted_buffers),
                batch_size=batch_size,
                shuffle=True,
            ))
        else:
            # Regular (non-transcoder) logic
            mixing_buffer = torch.cat([self.get_buffer(half_batches), self.storage_buffer], dim=0)
            mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]
            self.storage_buffer = mixing_buffer[:mixing_buffer.shape[0]//2]
            data_for_loader = mixing_buffer[mixing_buffer.shape[0]//2:]
            
            dataloader = iter(DataLoader(
                cast(Any, data_for_loader),
                batch_size=batch_size,
                shuffle=True,
            ))
        
        return dataloader

    def next_batch(self) -> torch.Tensor:
        """
        Get the next batch from the current DataLoader. If the DataLoader is exhausted,
        refill the buffer and create a new DataLoader, then fetch the next batch.
        """
        try:
            return F.normalize(next(self.dataloader), dim=-1)
        except StopIteration:
            print("Refilling dataloader")
            self.dataloader = self.get_data_loader()
            return F.normalize(next(self.dataloader), dim=-1)

