from datasets import load_dataset
import zstandard as zstd
import io
import json
import os
from nnsight import LanguageModel

from .trainers.top_k import AutoEncoderTopK
from .trainers.batch_top_k import BatchTopKSAE
from .trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
from .dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpReluAutoEncoder,
)

import torch

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size):
        super().__init__()

        sampler = torch.utils.data.RandomSampler(dataset)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=8,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError


def hf_dataset_to_generator(dataset_name, split="train", streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        for x in iter(dataset):
            yield x["text"]

    return gen()


def zst_to_generator(data_path):
    """
    Load a dataset from a .jsonl.zst file.
    The jsonl entries is assumed to have a 'text' field
    """
    compressed_file = open(data_path, "rb")
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(compressed_file)
    text_stream = io.TextIOWrapper(reader, encoding="utf-8")

    def generator():
        for line in text_stream:
            yield json.loads(line)["text"]

    return generator()


def get_nested_folders(path: str) -> list[str]:
    """
    Recursively get a list of folders that contain an ae.pt file, starting the search from the given path
    """
    folder_names = []

    for root, dirs, files in os.walk(path):
        if "ae.pt" in files:
            folder_names.append(root)

    return folder_names


def load_dictionary(base_path: str, device: str, step: int = None) -> tuple:
    if step:
        ae_path = f"{base_path}/checkpoints/ae_{step}.pt"
    else:
        ae_path = f"{base_path}/ae.pt"
    config_path = f"{base_path}/config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    dict_class = config["trainer"]["dict_class"]

    if dict_class == "AutoEncoder":
        dictionary = AutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "GatedAutoEncoder":
        dictionary = GatedAutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderNew":
        dictionary = AutoEncoderNew.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderTopK":
        k = config["trainer"]["k"]
        dictionary = AutoEncoderTopK.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "BatchTopKSAE":
        k = config["trainer"]["k"]
        dictionary = BatchTopKSAE.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "MatryoshkaBatchTopKSAE":
        k = config["trainer"]["k"]
        dictionary = MatryoshkaBatchTopKSAE.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "JumpReluAutoEncoder":
        dictionary = JumpReluAutoEncoder.from_pretrained(ae_path, device=device)
    else:
        raise ValueError(f"Dictionary class {dict_class} not supported")

    return dictionary, config


def get_submodule(model: LanguageModel, layer: int):
    """Gets the residual stream submodule"""
    model_name = model._model_key

    if "pythia" in model_name:
        return model.gpt_neox.layers[layer]
    elif "gemma" in model_name:
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")
