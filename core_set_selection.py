import glob
import logging
from scipy.stats import hmean
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from mixup_layer import MixupLayer
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from sklearn.metrics import adjusted_rand_score
from stft import STFT, FFT
from multi_resolution_net import STFT2dEncoderLayer, STFT1dEncoderLayer, FFTEncoderLayer
from dataset import SimpleDataset
import argparse
import sys
import random
import torchaudio

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--audioset_dir", type=str, required=True, help="Path to audioset."
)
parser.add_argument(
    "--model_dir", type=str, required=True, help="Name of model's dir."
)
parser.add_argument(
    "--segments", type=str, default="eval_segments", help="Name of audioset segment."
)
parser.add_argument("--seed", type=int, default=0, help="Seed")
parser.add_argument(
    "--use_att", type=str, default="wodo", help="Condition of model training."
)
args = parser.parse_args()
model_dir = args.model_dir
kmeans_mode = "original"
segments = args.segments
use_att = args.use_att
os.makedirs(model_dir, exist_ok=True)
config = {}
target_sr = 16000
use_bias = False
device = "cuda"
save_interval = 10
n_subclusters = 16
n_so_centroids = 16
audioset_dir = args.audioset_dir
attribute_dict = {
    "bandsaw": ["vel"],
    "bearing": ["vel", "loc"],
    "fan": ["m-n"],
    "gearbox": ["volt", "wt"],
    "grinder": ["grindstone", "plate"],
    "shaker": ["speed"],
    "slider": ["vel", "ac"],
    "ToyCar": ["car", "speed", "mic"],
    "ToyDrone": ["car", "speed", "mic"],
    "ToyNscale": ["car", "speed", "mic"],
    "ToyTank": ["car", "speed"],
    "ToyTrain": ["car", "speed", "mic"],
    "Vacuum": ["car", "mic"],
    "valve": ["pat"],
}

machines = sorted(attribute_dict.keys())
seed = args.seed

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logging.info(machines)
logging.info(f"model_dir:{model_dir}")
check_path = f"{model_dir}/audio_{segments}.csv"
if os.path.exists(check_path):
    logging.info(f"{check_path} is exists! Program is broken")
    import sys

    sys.exit()


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed)


def count_params(model):
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    return params

def length_norm_torch(mat):
    norms = torch.norm(mat, dim=1, keepdim=True)
    return mat / norms


class ModelEmbCNN(nn.Module):
    def __init__(self, use_bias=False):
        super(ModelEmbCNN, self).__init__()
        self.mixup_layer = MixupLayer(prob=0.5)
        ################################################################
        sec = 18
        sr = 16000
        emb_base_size = 128
        se_mode = "normal"
        stft_cfg_list = [
            {
                "sr": sr,
                "n_fft": 4096,
                "hop_length": 2048,
                "n_mels": None,
                "power": 1.0,
                "use_mel": False,
                "f_min": 200.0,
                "f_max": 8000.0,
                "temporal_norm": False,
            },
            {
                "sr": sr,
                "n_fft": 128,
                "hop_length": 64,
                "n_mels": None,
                "power": 1.0,
                "use_mel": False,
                "f_min": 200.0,
                "f_max": 8000.0,
                "temporal_norm": False,
            },
        ]
        # STFT #########################################################
        self.stft_layer_list = nn.ModuleList([])
        for stft_cfg in stft_cfg_list:
            stft = STFT(**stft_cfg)
            spectrogram_size = stft(torch.randn(sec * sr)).shape
            if min(spectrogram_size) >= 36:
                stft_encoder = STFT2dEncoderLayer(
                    spectrogram_size, use_bias, emb_base_size, se_mode
                )
            elif spectrogram_size[0] >= 36 and spectrogram_size[1] < 36:
                stft_encoder = STFT1dEncoderLayer(
                    spectrogram_size, 1, use_bias, emb_base_size
                )
            elif spectrogram_size[0] < 36 and spectrogram_size[1] >= 36:
                stft_encoder = STFT1dEncoderLayer(
                    spectrogram_size, 2, use_bias, emb_base_size
                )
            else:
                raise ValueError("input sequence is too short")
            self.stft_layer_list.append(nn.Sequential(stft, stft_encoder))
        # FFT ##########################################################
        self.fft_layer = nn.Sequential(
            FFT(), FFTEncoderLayer(sec, sr, use_bias, emb_base_size)
        )

    def forward(self, x, y, use_mixup=False):
        """フォワード.
        x: (B, L)
        y: (B, N_class) のワンホットベクトル
        """
        # Mixup
        self.mixup_layer.training = self.training * use_mixup
        x_mix, y_mix = self.mixup_layer(x, y)
        # processed by network
        emb_list = [self.fft_layer(x_mix)]
        for stft_layer in self.stft_layer_list:
            emb_list += [stft_layer(x_mix)]
        emb = torch.cat(emb_list, dim=1)
        return emb, y_mix, emb_list



def get_score_df(df):
    dev_cols = [
        "bearing",
        "fan",
        "gearbox",
        "slider",
        "ToyCar",
        "ToyTrain",
        "valve",
    ]
    eval_cols = [
        "bandsaw",
        "grinder",
        "shaker",
        "ToyDrone",
        "ToyNscale",
        "ToyTank",
        "Vacuum",
    ]


    index = [
        "source_auc",
        "source_pauc",
        "target_auc",
        "target_pauc",
        "all_auc",
        "all_pauc",
        "in_source_auc",
        "in_target_auc",
        "official_score",
    ]
    score_df = pd.DataFrame(
        index=index, columns=dev_cols + eval_cols + ["dev", "eval", "all"]
    )
    for machine in sorted(df.machine.unique(), reverse=True):
        official_score = []
        idx = df["machine"] == machine
        y_pred = df.loc[idx, "score"].to_numpy()
        y_true = 1 - df.loc[idx, "is_normal"].to_numpy()
        auc = roc_auc_score(y_true, y_pred)
        score_df.loc["all_auc", machine] = auc
        pauc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
        score_df.loc["all_pauc", machine] = pauc
        official_score.append(pauc)
        for domain in ["source", "target"]:
            idx = (df["machine"] == machine) & (df["domain"] == domain)
            y_pred = df.loc[idx, "score"].to_numpy()
            y_true = 1 - df.loc[idx, "is_normal"].to_numpy()
            auc = roc_auc_score(y_true, y_pred)
            score_df.loc[f"in_{domain}_auc", machine] = auc
            idx = idx | (df["machine"] == machine) & (~df["is_normal"])
            y_pred = df.loc[idx, "score"].to_numpy()
            y_true = 1 - df.loc[idx, "is_normal"].to_numpy()
            auc = roc_auc_score(y_true, y_pred)
            score_df.loc[f"{domain}_auc", machine] = auc
            official_score.append(auc)
            pauc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
            score_df.loc[f"{domain}_pauc", machine] = pauc
        score_df.loc["official_score", machine] = hmean(official_score)
    for idx in index:
        score_df.loc[idx, "dev"] = hmean(score_df.loc[idx, dev_cols])
        score_df.loc[idx, "eval"] = hmean(score_df.loc[idx, eval_cols])
        score_df.loc[idx, "all"] = hmean(score_df.loc[idx, dev_cols + eval_cols])
    if "pred_cluster" in df.columns:
        ari_score_list = []
        for machine in dev_cols + eval_cols:
            idx = (df["machine"] == machine) & df["is_normal"]
            ari_score = adjusted_rand_score(
                df.loc[idx, "att"], df.loc[idx, "pred_cluster"]
            )
            score_df.loc["ARI", machine] = ari_score
            ari_score_list.append(ari_score)
        avg_ari = sum(ari_score_list) / len(ari_score_list)
        score_df.loc["ARI", "all"] = avg_ari
        score_df.loc["ARI", "dev"] = score_df.loc["ARI", dev_cols].mean()
        score_df.loc["ARI", "eval"] = score_df.loc["ARI", eval_cols].mean()
    return score_df


# encode ids as labels
# get dataset
df_list = []
columns = ["path", "fname", "domain", "phase", "state", "machine", "att"]
df_list = []
for machine in machines:
    train_list = sorted(glob.glob(f"dev_data/{machine}/train/*.wav"))
    test_list = sorted(glob.glob(f"dev_data/{machine}/test/*.wav"))
    path_list = train_list + test_list
    tmp_df = pd.DataFrame(path_list, columns=["path"])
    tmp_df["machine"] = machine
    df_list.append(tmp_df)
df = pd.concat(df_list, axis=0).reset_index(drop=True)
df["fname"] = df["path"].map(lambda x: x.split("/")[-1])
df["domain"] = df["fname"].map(lambda x: x.split("_")[2])
df["phase"] = df["fname"].map(lambda x: x.split("_")[3])
df["att"] = df["fname"].map(lambda x: "".join(x.split("_")[6:]).replace(".wav", ""))
df["state"] = df["fname"].map(lambda x: x.split("_")[4])
df["is_normal"] = df["state"] == "normal"
train_df = df[df["phase"] != "test"].reset_index(drop=True)
train_files = train_df["path"].to_numpy()
train_atts = train_df["att"].to_numpy()
train_ids = train_df["machine"].to_numpy()
train_domains = train_df["domain"].to_numpy()
train_normal = train_df["is_normal"].to_numpy()

eval_df = df[df["phase"] == "test"].reset_index(drop=True)
eval_files = eval_df["path"].to_numpy()
eval_atts = eval_df["att"].to_numpy()
eval_ids = eval_df["machine"].to_numpy()
eval_domains = eval_df["domain"].to_numpy()
eval_normal = eval_df["is_normal"].to_numpy()
le_4train = LabelEncoder()
source_train = np.array(
    [file.split("_")[3] == "source" for file in train_files.tolist()]
)
source_eval = np.array([file.split("_")[3] == "source" for file in eval_files.tolist()])
use_att = "wodo"
if use_att == "oracle":
    train_ids_4train = np.array(
        [
            "###".join([train_ids[k], train_atts[k]])
            for k in np.arange(train_ids.shape[0])
        ]
    )
    eval_ids_4train = np.array(
        ["###".join([eval_ids[k], eval_atts[k]]) for k in np.arange(eval_ids.shape[0])]
    )
elif use_att == "wodo":
    train_ids_4train = np.array(
        ["###".join([train_ids[k]]) for k in np.arange(train_ids.shape[0])]
    )
    eval_ids_4train = np.array(
        ["###".join([eval_ids[k]]) for k in np.arange(eval_ids.shape[0])]
    )
le_4train.fit(np.concatenate([train_ids_4train, eval_ids_4train], axis=0))
num_classes_4train = len(
    np.unique(np.concatenate([train_ids_4train, eval_ids_4train], axis=0))
)
train_labels_4train = le_4train.transform(train_ids_4train)
eval_labels_4train = le_4train.transform(eval_ids_4train)
train_df["label"] = train_labels_4train
eval_df["label"] = eval_labels_4train

le = LabelEncoder()
train_labels = le.fit_transform(train_ids)
eval_labels = le.transform(eval_ids)
train_df["machine_id"] = train_labels
eval_df["machine_id"] = eval_labels
print(num_classes_4train, "num_classes_4train", train_labels, "train_labels")


model_path = os.path.join(model_dir, "model.pt")
state_dict = torch.load(model_path, map_location="cpu")
model = ModelEmbCNN(use_bias=use_bias)
model.to(device)
params_cnt = count_params(model)
logging.info(f"Size of model is {params_cnt}.")
emb_size = 384
model.load_state_dict(state_dict["model"])
model.to(device)
model.eval()

centroid_path = f"{model_dir}/centroids.npy"
eval_columns = [
    "path",
    "is_normal",
    "domain",
    "machine",
    "att",
    "score",
    "pred_cluster",
]
train_loaders = {}
eval_loaders = {}
for machine in machines:
    train_dataset = SimpleDataset(
        train_df[train_df["machine"] == machine],
        allow_cache=False,
        use_adjust_size=True,
    )
    train_loaders[machine] = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1000,
        num_workers=1,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )
    eval_dataset = SimpleDataset(
        eval_df[eval_df["machine"] == machine],
        allow_cache=False,
        use_adjust_size=True,
    )
    eval_loaders[machine] = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=200,
        num_workers=1,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )

epoch_eval_df_list = []
centroids = np.zeros((len(machines) * (16 + 10), emb_size))
for ii, machine in enumerate(machines):
    train_loader = train_loaders[machine]
    eval_loader = eval_loaders[machine]
    # lab = le.transform([machine])[0]
    for batch in train_loader:
        with torch.no_grad():
            train_domains = np.array(batch[3])
            x = batch[0].to(device)
            emb, _, _ = model(x, None)
            x_train_ln = length_norm_torch(emb).cpu().numpy()
            logging.info(f"x: {x.shape} {machine} {x_train_ln.shape}")
    for batch in eval_loader:
        with torch.no_grad():
            x = batch[0].to(device)
            emb, _, _ = model(x, None)
            x_eval_ln = length_norm_torch(emb).cpu().numpy()
            logging.info(f"x: {x.shape} {machine} {x_eval_ln.shape}")
    so_kmeans = KMeans(n_clusters=16, random_state=seed).fit(
        x_train_ln[train_domains == "source"]
    )
    means_source_ln = so_kmeans.cluster_centers_
    means_target_ln = x_train_ln[train_domains == "target"]
    centroids[ii * 26 : 26 * (ii + 1)] = np.concatenate(
        [means_source_ln, means_target_ln], axis=0
    )
    eval_cos = np.min(
        1 - np.dot(x_eval_ln, means_target_ln.transpose()),
        axis=-1,
        keepdims=True,
    )
    eval_cos = np.minimum(
        eval_cos,
        np.min(
            1 - np.dot(x_eval_ln, means_source_ln.transpose()),
            axis=-1,
            keepdims=True,
        ),
    )
    eval_dist = np.min(eval_cos, axis=-1)
    epoch_eval_df = eval_df[eval_df["machine"] == machine].reset_index(drop=True)
    epoch_eval_df[eval_columns[5]] = eval_dist
    epoch_eval_df_list.append(epoch_eval_df)
epoch_eval_df = pd.concat(epoch_eval_df_list)
epoch_eval_df[eval_columns[:6]].to_csv(
    f"{model_dir}/eval_{kmeans_mode}.csv", index=False
)
score_df = get_score_df(epoch_eval_df) * 100
score_df.to_csv(f"{model_dir}/score_for_audio_classfiy.csv")
np.save(centroid_path, centroids)


def adjust_size(wav, new_size=288000):
    reps = int(np.ceil(new_size / wav.shape[0]))
    offset = torch.randint(
        low=0, high=int(reps * wav.shape[0] - new_size + 1), size=(1,)
    ).item()
    return wav.repeat(reps)[offset : offset + new_size]


class AudioDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        path_list,
    ):
        """Initialize dataset."""
        self.path_list = path_list

    def __getitem__(self, idx):
        wave, _ = torchaudio.load(self.path_list[idx])
        wave = adjust_size(wave.squeeze())
        return self.path_list[idx], wave

    def __len__(self):
        """Return dataset length."""
        return len(self.path_list)


if "unbalanced_train_segments" == segments:
    audioset_list = list(
        glob.glob(
            f"{audioset_dir}/sr16_audios/unbalanced_train_segments/**/*.wav",
            recursive=True,
        )
    )
else:
    audioset_list_path = f"{audioset_dir}/sr16_audios/{segments}/*.wav"
    audioset_list = list(glob.glob(audioset_list_path))
logging.info(f"audioset_list:{len(audioset_list)}")
dataset = AudioDataset(audioset_list)
batch_size = 1000
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1000,
    num_workers=1,
    pin_memory=False,
    shuffle=False,
    drop_last=False,
)


def calculate_similarity_and_argmin(e1, e2):
    dot_product = torch.matmul(e2, e1.transpose(0, 1))
    result = 1 - dot_product
    argmin_index = torch.argmin(result)
    return result, argmin_index


centroids = torch.tensor(centroids).float().to(device)
anomaly_score_list = np.zeros((len(audioset_list), len(machines)))
idx_list = []
for i in range(len(machines)):
    machine_idx = np.zeros(26 * len(machines))
    machine_idx[i * 26 : (i + 1) * 26] = 1
    idx_list.append(machine_idx.astype(bool))
cnt = 0
for n_iter, batch in enumerate(tqdm(loader)):
    cnt += batch_size
    with torch.no_grad():
        emb, _, _ = model(batch[1].to(device), None)
        audio_emb = length_norm_torch(emb)
        anomaly_score = (
            1
            - torch.matmul(centroids, audio_emb.transpose(0, 1))
            .transpose(0, 1)
            .cpu()
            .numpy()
        )  # (26*14, 384) x (384, B) = (26*14, B) -> (B, 26*14)
        for idx in range(len(machines)):
            anomaly_score_list[
                n_iter * batch_size : (n_iter + 1) * batch_size, idx
            ] += anomaly_score[:, idx_list[idx]].min(axis=1)
        logging.info(f"audioset: {cnt}")

audio_score_df = pd.DataFrame(anomaly_score_list, columns=machines)
audio_score_df["path"] = audioset_list
audio_score_df["fname"] = audio_score_df["path"].map(
    lambda x: x.split("/")[-1].split(".")[0]
)

file_path = f"{audioset_dir}/metadata/{segments}.csv"
with open(file_path, "r") as f:
    lines = f.readlines()
lines = [line.strip() for line in lines if not line.startswith("#")]
data = []
for line in lines:
    parts = line.split(",")
    ytid = "Y" + parts[0]
    start_seconds = parts[1]
    end_seconds = parts[2]
    positive_labels = (
        parts[3].replace('"', "").replace("\t", "").replace(" ", "").split(",")
    )
    for label in positive_labels:
        data.append([ytid, start_seconds, end_seconds, label])

df = pd.DataFrame(data, columns=["fname", "start_seconds", "end_seconds", "mid"])
use_col = ["fname", "mid"]
audio_score_df = audio_score_df.merge(df[use_col], on="fname", how="left")
print(audio_score_df.shape)
audio_score_df.to_csv(f"{model_dir}/audio_{segments}.csv", index=False)
