from multiprocessing import Manager
import random
import numpy as np
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset
import torchaudio
import torch


def adjust_size(wav, new_size=288000):
    reps = int(np.ceil(new_size / wav.shape[0]))
    offset = torch.randint(
        low=0, high=int(reps * wav.shape[0] - new_size + 1), size=(1,)
    ).item()
    return wav.repeat(reps)[offset: offset + new_size]


class SimpleDataset(Dataset):

    def __init__(
        self,
        df,
        allow_cache=False,
        use_sampler=False,
        use_adjust_size=False,
        audioset_list=[],
    ):
        self.use_adjust_size = use_adjust_size
        self.audioset_list = audioset_list
        if use_sampler:
            so_df = df[df["domain"]=="source"]
            ta_df = df[df["domain"]=="target"]
            self.so_files = so_df.loc[:, "path"].tolist()
            self.ta_files = ta_df.loc[:, "path"].tolist()
            self.path_list = self.so_files + self.ta_files
            self.labels = so_df.loc[:, "label"].tolist() + ta_df.loc[:, "label"].tolist()
            self.atts = so_df.loc[:, "att"].astype(str).tolist() + ta_df.loc[:, "att"].astype(str).tolist()
            self.machine_ids = so_df.loc[:, "machine_id"].tolist() + ta_df.loc[:, "machine_id"].tolist()
            self.domains = so_df.loc[:, "domain"].tolist() + ta_df.loc[:, "domain"].tolist()
            self.is_normal = so_df.loc[:, "is_normal"].tolist() + ta_df.loc[:, "is_normal"].tolist()
            self.machine = so_df.loc[:, "machine"].tolist() + ta_df.loc[:, "machine"].tolist()
        else:
            self.path_list = df.loc[:, "path"].tolist()
            self.labels = df.loc[:, "label"].tolist()
            self.atts = df.loc[:, "att"].astype(str).tolist()
            self.machine_ids = df.loc[:, "machine_id"].tolist()
            self.domains = df.loc[:, "domain"].tolist()
            self.is_normal = df.loc[:, "is_normal"].tolist()
            self.machine = df.loc[:, "machine"].tolist()
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE: Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(df))]

    def __getitem__(self, idx):
        """Get specified idx items."""
        if self.allow_cache:
            if len(self.caches[idx]) != 0:
                wave = self.caches[idx]
            else:
                wave, _ = torchaudio.load(self.path_list[idx])
                wave = wave.squeeze()
        else:
            wave, _ = torchaudio.load(self.path_list[idx])
            wave = wave.squeeze()
        if self.allow_cache and len(self.caches[idx]) == 0:
            self.caches[idx] = wave
        if self.use_adjust_size:
            wave = adjust_size(wave)
        if len(self.audioset_list) > 0:
            audio_path = random.choice(self.audioset_list)
            noise, sf = torchaudio.load(audio_path)
            noise = adjust_size(noise.squeeze())
            return wave, self.labels[idx], self.machine_ids[idx], self.domains[idx], self.atts[idx], self.is_normal[idx], noise
        return wave, self.labels[idx], self.machine_ids[idx], self.domains[idx], self.atts[idx], self.is_normal[idx]

    def __len__(self):
        """Return dataset length."""
        return len(self.path_list)


class DomainBatchSampler(BatchSampler):

    def __init__(
        self,
        dataset,
        n_so=32,
        n_ta=32,
        shuffle=False,
        drop_last=False,
    ):
        """Batch Sampler.

        Args:
            dataset (dataset): dataset for ASD
            n_so (int, optional): The number of positive sample in the mini-batch. Defaults to 32.
            n_ta (int, optional): The number of negative sample in the mini-batch. Defaults to 32.
            shuffle (bool, optional): Shuffle. Defaults to False.
            drop_last (bool, optional): Drop last. Defaults to False.
        """
        self.n_so_file = len(dataset.so_files)
        self.n_ta_file = len(dataset.ta_files)
        self.so_idx = np.arange(self.n_so_file)
        self.ta_idx = np.arange(
            self.n_so_file,
            self.n_so_file + self.n_ta_file,
        )
        self.used_idx_cnt = {"so": 0}
        self.count = 0
        self.n_so = n_so
        self.n_ta = n_ta
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        self.count = 0
        if self.shuffle:
            np.random.shuffle(self.so_idx)
            np.random.shuffle(self.ta_idx)
        while self.count < self.n_so_file:
            indices = []
            indices.extend(
                self.so_idx[
                    self.used_idx_cnt["so"] : self.used_idx_cnt["so"] + self.n_so
                ]
            )
            self.used_idx_cnt["so"] += self.n_so
            indices.extend(
                random.choices(self.ta_idx, k=self.n_ta)
            )
            if self.shuffle:
                random.shuffle(indices)
            yield indices
            self.count += self.n_so
            if self.count + self.n_so < self.n_so_file:
                self.count += self.n_so

        if not self.drop_last:
            indices = []
            indices.extend(self.so_idx[self.used_idx_cnt["so"] :])
            indices.extend(random.choices(self.ta_idx, k=self.n_ta))
            yield indices
        self.used_idx_cnt["so"] = 0

    def __len__(self):
        return self.n_so_file // self.n_so + 1
