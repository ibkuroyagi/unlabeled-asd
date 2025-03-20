from torchaudio.transforms import PitchShift
import logging
from scipy.stats import hmean
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from mixup_layer import MixupLayer
from subcluster_adacos import SCAdaCos
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict
import matplotlib.pyplot as plt
from dataset import SimpleDataset
from dataset import DomainBatchSampler
import seaborn as sns
from stft import STFT, FFT
from multi_resolution_net import STFT2dEncoderLayer, STFT1dEncoderLayer, FFTEncoderLayer
import argparse
import umap
import sys
import random

parser = argparse.ArgumentParser(description="")
parser.add_argument("--tag", type=str, required=True, help="Save dir name.")
parser.add_argument(
    "--use_att", type=str, default="wodo", help="Name of positive machine."
)
parser.add_argument("--seed", type=int, default=0, help="Seed")
parser.add_argument("--n_max", type=int, default=1000, help="Maximum number of audioset per machine type.")
parser.add_argument("--epochs", type=int, default=50, help="epoch")
parser.add_argument("--dumy_dir", type=str, default="", help="pre trained model's output dir for pseudo labels.")
parser.add_argument("--audio_dir", type=str, default="", help="pre trained model's output dir for core set external data.")
args = parser.parse_args()
tag = args.tag
use_att = args.use_att
kmeans_mode = "original"
metric = "cosine"
n_max = args.n_max
os.makedirs(f"{tag}", exist_ok=True)
config = {}
target_sr = 16000
use_mixup = True
use_bias = False
device = "cuda"
so_size = 80
ta_size = 20
batch_size = so_size + ta_size
epochs = args.epochs
save_interval = 10
n_subclusters = 16
n_so_centroids = 16
n_ta_centroids = 4
dumy_dir = args.dumy_dir
audio_dir = args.audio_dir
mixup_prob = 0.5
margin = 0.5
lr = 1e-3
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
machines = list(attribute_dict.keys())

seed = args.seed

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logging.info(f"tag:{tag}, use_att:{use_att}")


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


def length_norm(mat):
    # Calculate the L2 norm for each line
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
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
        """
        x: (B, L)
        y: (B, N_class)
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


def random_int_exclude_zero():
    possible_values = [-12, -11, -10, -9, -8, -7, -6, 6, 7, 8, 9, 10, 11, 12]
    return random.choice(possible_values)


df_list = []
columns = ["path", "fname", "domain", "phase", "state", "machine", "att"]
for machine, att_list in attribute_dict.items():
    tmp_df = pd.read_csv(
        f"../asd_dcase2023/scripts/downloads/{machine}/attributes_0.csv"
    )
    tmp_df["att"] = ""
    for att in att_list:
        tmp_df["att"] = tmp_df["att"] + tmp_df[att].astype(str) + "_"
    df_list.append(tmp_df[columns])
df = pd.concat(df_list, axis=0).reset_index(drop=True)
df["path"] = df["path"].map(lambda x: x.replace("downloads", "dev_data"))
df["is_normal"] = df["state"] == "normal"

train_df = df[df["phase"] != "test"].reset_index(drop=True)
#######################
if n_max > 0:
    train_score_df = pd.read_csv(f"{audio_dir}/train_original.csv")
    df_list = []
    for segments in [
        "balanced_train_segments",
        "eval_segments",
        "unbalanced_train_segments",
    ]:
        audio_path = f"{audio_dir}/audio_{segments}.csv"
        tmp_df = pd.read_csv(audio_path)
        df_list.append(tmp_df)
    audio_df = pd.concat(df_list)
    use_audio_df_list = []
    for machine in machines:
        thredhold = train_score_df.loc[
            train_score_df["machine"] == machine, "score"
        ].max()
        use_audio_df = audio_df.loc[
            audio_df[machine] <= thredhold,
            ["path", "fname", "mid", machine],
        ].reset_index(drop=True)
        use_audio_df = use_audio_df.rename(columns={machine: "score", "mid": "att"})
        use_audio_df["machine"] = "audio_" + machine
        # logging.info(machine, len(use_audio_df))
        use_audio_df_list.append(use_audio_df)
    use_audio_df = pd.concat(use_audio_df_list).reset_index(drop=True)
    logging.info(use_audio_df.shape)
    machine_counts = use_audio_df["machine"].value_counts()

    machine_ranks = machine_counts.rank(method="first").to_dict()
    logging.info(use_audio_df["machine"].value_counts())

    duplicated_paths = use_audio_df["path"][use_audio_df["path"].duplicated(keep=False)]

    duplicated_df = use_audio_df.loc[duplicated_paths.index]

    duplicated_df["machine_score"] = duplicated_df["machine"].map(machine_ranks)

    selected_rows = duplicated_df.loc[
        duplicated_df.groupby("path")["machine_score"].idxmin()
    ]

    non_duplicated_df = use_audio_df.drop(duplicated_paths.index)
    use_audio_df = (
        pd.concat([non_duplicated_df, selected_rows])
        .drop(columns=["machine_score"])
        .reset_index(drop=True)
    )
    logging.info(use_audio_df.shape)
    logging.info(use_audio_df["machine"].value_counts())
    sorted_df = use_audio_df.sort_values(by=["machine", "score"])
    audio_df = sorted_df.groupby("machine").head(n_max).reset_index(drop=True)
    audio_df["domain"] = "source"
    audio_df["phase"] = "train"
    audio_df["state"] = "normal"
    audio_df["is_normal"] = True
    cnt = audio_df[["machine", "att"]].value_counts()
    logging.info(f"n_max {n_max}, len(cnt):{len(cnt)} audio_df:{audio_df.shape}")
    train_df = pd.concat([train_df, audio_df[train_df.columns]])
######################
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
source_train = train_domains == "source"
source_eval = np.array([file.split("_")[3] == "source" for file in eval_files.tolist()])
if use_att == "dumy":
    dumy_path = dumy_dir + "/train_original.csv"
    tmp_df = pd.read_csv(dumy_path)
    logging.info(dumy_path)
    train_df = train_df.merge(tmp_df[["path", "pred_cluster"]], on="path", how="left")
    idx = train_df[
        "pred_cluster"
    ].isna() 
    logging.info(f"train_df:{train_df.shape}, idx:{idx.sum()}")
    train_df["pred_cluster"] = train_df["pred_cluster"].astype(str)
    train_df.loc[idx, "pred_cluster"] = train_df.loc[idx, "att"]
    logging.info(train_df["pred_cluster"])
    train_dumy_atts = train_df["pred_cluster"].to_numpy()
    train_ids_4train = np.array(
        [
            "###".join([train_ids[k], train_dumy_atts[k], str(source_train[k])])
            for k in np.arange(train_ids.shape[0])
        ]
    )
    eval_ids_4train = np.array(
        [
            "###".join([eval_ids[k], str(source_eval[k])])
            for k in np.arange(eval_ids.shape[0])
        ]
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
logging.info(f"{num_classes_4train}, num_classes_4train, {train_labels}, train_labels")


def cosine_similarity_matrix(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed_vectors = vectors / norms
    sim_matrix = np.dot(normed_vectors, normed_vectors.T)
    return sim_matrix


def mix_signals(x1, x2, snr_min, snr_max):
    snr_dB = torch.empty(x1.size(0)).uniform_(snr_min, snr_max).to(device)
    snr_linear = 10 ** (snr_dB / 20)
    power_x1 = torch.sum(x1**2, dim=1) / x1.size(1)
    power_x2 = torch.sum(x2**2, dim=1) / x1.size(1)
    scaling_factor = torch.sqrt(power_x1 / (power_x2 * snr_linear))
    scaling_factor[power_x2 == 0] = 0
    x2_scaled = x2 * scaling_factor.unsqueeze(1)
    mixed_signal = x1 + x2_scaled
    return mixed_signal, snr_dB


def distance(e1, e2, metric="cosine"):
    """
    Args:
        e1, e2 (Tensor): embeddings (B, D)
    """
    if metric == "euclid":
        return torch.sqrt(torch.sum((e1 - e2) ** 2, dim=1))
    elif metric == "cosine":
        e1 = F.normalize(e1, dim=1)
        e2 = F.normalize(e2, dim=1)
        return (1 - torch.sum(e1 * e2, dim=1)) / 0.2


train_dataset = SimpleDataset(
    train_df,
    allow_cache=n_max < 5000,
    use_sampler=True,
    use_adjust_size=True,
    audioset_list=[],
)
sampler = DomainBatchSampler(
    train_dataset,
    n_so=so_size,
    n_ta=ta_size,
    shuffle=True,
    drop_last=True,
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_sampler=sampler,
    num_workers=1,
    pin_memory=True,
)
train_loaders = {}
eval_loaders = {}
for machine in machines:
    train_dataset = SimpleDataset(
        train_df[train_df["machine"] == machine], allow_cache=True, use_adjust_size=True
    )
    train_loaders[machine] = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1000,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    eval_dataset = SimpleDataset(
        eval_df[eval_df["machine"] == machine], allow_cache=True, use_adjust_size=True
    )
    eval_loaders[machine] = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=200,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
model = ModelEmbCNN(use_bias=use_bias)
model.to(device)
params_cnt = count_params(model)
logging.info(f"Size of model is {params_cnt}.")
emb_size = 384
simple_scadacos = SCAdaCos(
    n_classes=num_classes_4train,
    n_subclusters=n_subclusters,
    trainable=False,
    emb_size=emb_size,
).to(device)
params_list = [
    {"params": model.parameters()},
    {"params": simple_scadacos.parameters()},
]

mrml_dict = {}
for i in range(3):
    mrml_dict[i] = SCAdaCos(
        n_classes=num_classes_4train,
        n_subclusters=n_subclusters,
        trainable=True,
        emb_size=128,
    ).to(device)
    params_list.append({"params": mrml_dict[i].parameters()})
optimizer = torch.optim.AdamW(params_list, lr=lr)

steps = 0
columns = ["path", "is_normal", "domain", "machine", "att", "score", "pred_cluster"]
check_path = os.path.join(f"{tag}/epoch{epochs}", "model.pt")

for epoch in range(1, epochs + 1):
    if os.path.exists(check_path):
        logging.info(f"{check_path} is exists!")
        break
    total_train_loss = defaultdict(float)
    epoch_train_embeds = np.empty((0, emb_size))
    epoch_train_machine_ids = np.empty(0)
    epoch_train_domains = np.empty(0)
    epoch_train_atts = np.empty(0)

    total_eval_loss = defaultdict(float)
    epoch_eval_embeds = np.empty((0, emb_size))
    epoch_eval_machine_ids = np.empty(0)
    epoch_eval_domains = np.empty(0)
    epoch_eval_atts = np.empty(0)
    epoch_is_normal = np.empty(0)
    # train steps
    for steps_per_epoch, batch in enumerate(train_loader):
        model.train()
        steps += 1
        loss = 0
        logging.info(f"[Train] Epoch:{epoch}, Step:{steps}")
        anc = batch[0].to(device)
        y = F.one_hot(batch[1], num_classes=num_classes_4train).float().to(device)
        anc_out = model(anc, y, use_mixup=use_mixup)
        loss1 = simple_scadacos(anc_out[0], anc_out[1])
        if loss1.item() > 0:
            loss += loss1
        total_train_loss["train/simple_scadacos"] += loss1.item()
        logging.info(f"loss1: simple_scadacos:{loss1.item():.2f}")
        loss2 = 0
        for key, mrml_scadacos in mrml_dict.items():
            loss2 += mrml_scadacos(anc_out[2][key], anc_out[1])
        if loss2.item() > 0:
            loss += loss2
        total_train_loss["train/mrml_scadacos"] += loss2.item()
        logging.info(f"loss2: mrml_scadacos:{loss2.item():.2f}")
        loss3 = 0
        anc_out = model(anc, None, use_mixup=False)
        machine_ids = batch[2].to(device)
        shuffle_idx = torch.randperm(len(anc)).to(device)
        shuffle_machine_ids = machine_ids[shuffle_idx]
        is_same_machine_idx = shuffle_machine_ids == machine_ids
        noise = anc[shuffle_idx]
        noise[~is_same_machine_idx] = torch.zeros_like(
            noise[~is_same_machine_idx]
        ).to(device)
        pos, _ = mix_signals(
            batch[0].to(device), noise, -5, 20
        )
        pos_emb, _, _ = model(pos, None, use_mixup=False)
        pitch_shift = PitchShift(
            sample_rate=16000,
            n_steps=random_int_exclude_zero(),
        ).to(device)
        neg = pitch_shift(anc)
        neg_emb, _, _ = model(neg, None, use_mixup=False)
        same_anc_emb = anc_out[0][is_same_machine_idx]
        pull_loss = distance(
            same_anc_emb, pos_emb[is_same_machine_idx], metric=metric
        )
        push_loss = distance(
            same_anc_emb, neg_emb[is_same_machine_idx], metric=metric
        )
        main_loss = torch.maximum(
            pull_loss - push_loss + margin,
            torch.zeros(len(same_anc_emb)).to(device),
        )
        loss3 = main_loss.sum()
        loss += loss3
        total_train_loss["train/triplet_loss"] += loss3.item()
        logging.info(f"loss3: triplet_loss:{loss3.item():.2f}")
        total_train_loss["train/loss"] += loss.item()
        logging.info(f"loss:{loss.item():.2f}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if epoch % save_interval == 0:
        model.eval()
        model.to("cpu")
        model_dir = f"{tag}/epoch{epoch}"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pt")
        state_dict = {
            "model": model.state_dict(),
        }
        torch.save(state_dict, model_path)
        logging.info(f"Successfully saved at {model_path}")
        model.to(device)

for epoch in range(save_interval, epochs + save_interval, save_interval):
    model_dir = f"{tag}/epoch{epoch}"
    model_path = os.path.join(model_dir, "model.pt")
    check_path = f"{model_dir}/score_{kmeans_mode}.csv"
    if os.path.exists(check_path):
        logging.info(f"{check_path} is exists! continue")
        continue
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    model.to(device)
    model.eval()
    # length normalization
    all_target = 10
    centroids = np.zeros((len(machines), n_so_centroids + all_target, emb_size))
    epoch_train_df_list = []
    epoch_eval_df_list = []
    for lab, machine in enumerate(machines):
        epoch_train_df = train_df[train_df["machine"] == machine].reset_index(drop=True)
        epoch_eval_df = eval_df[eval_df["machine"] == machine].reset_index(drop=True)
        train_loader = train_loaders[machine]
        eval_loader = eval_loaders[machine]
        # lab = le.transform([machine])[0]
        for batch in train_loader:
            with torch.no_grad():
                train_domains = np.array(batch[3])
                x = batch[0].to(device)
                emb, _, _ = model(x, None)
                x_train_ln = length_norm(emb.to("cpu").numpy())
                logging.info(f"x: {x.shape} {machine} {x_train_ln.shape}")
        for batch in eval_loader:
            with torch.no_grad():
                x = batch[0].to(device)
                emb, _, _ = model(x, None)
                x_eval_ln = length_norm(emb.to("cpu").numpy())
                logging.info(f"x: {x.shape} {machine} {x_eval_ln.shape}")
        so_kmeans = KMeans(n_clusters=n_so_centroids, random_state=seed).fit(
            x_train_ln[train_domains == "source"]
        )
        means_source_ln = so_kmeans.cluster_centers_
        means_target_ln = x_train_ln[train_domains == "target"]
        ta_kmeans = KMeans(n_clusters=n_ta_centroids, random_state=seed).fit(
            x_train_ln[train_domains == "target"]
        )
        # means_target_ln = ta_kmeans.cluster_centers_
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
        # UMAP
        x_ln = np.concatenate([x_train_ln, x_eval_ln, means_source_ln, means_target_ln])
        umap_model = umap.UMAP(random_state=seed, metric="cosine", n_neighbors=30)
        u = umap_model.fit_transform(x_ln)
        df1 = epoch_train_df[["att", "is_normal", "domain"]].reset_index(drop=True)
        df1["u0"] = u[: len(epoch_train_df), 0]
        df1["u1"] = u[: len(epoch_train_df), 1]
        df2 = epoch_eval_df[["att", "is_normal", "domain"]].reset_index(drop=True)
        df2["u0"] = u[len(epoch_train_df) : len(epoch_train_df) + len(epoch_eval_df), 0]
        df2["u1"] = u[len(epoch_train_df) : len(epoch_train_df) + len(epoch_eval_df), 1]
        df3 = pd.DataFrame(
            {
                "att": [f"so_ce{i}" for i in range(n_so_centroids)],
                "is_normal": [True for _ in range(n_so_centroids)],
                "domain": ["ce_source" for _ in range(n_so_centroids)],
                "u0": u[
                    len(epoch_train_df) + len(epoch_eval_df) : len(epoch_train_df)
                    + len(epoch_eval_df)
                    + n_so_centroids,
                    0,
                ],
                "u1": u[
                    len(epoch_train_df) + len(epoch_eval_df) : len(epoch_train_df)
                    + len(epoch_eval_df)
                    + n_so_centroids,
                    1,
                ],
            }
        )
        df4 = pd.DataFrame(
            {
                "att": [f"ta_ce{i+n_so_centroids}" for i in range(n_ta_centroids)],
                "is_normal": [True for _ in range(n_ta_centroids)],
                "domain": ["ce_target" for _ in range(n_ta_centroids)],
                "u0": u[-n_ta_centroids:, 0],
                "u1": u[-n_ta_centroids:, 1],
            }
        )
        umap_df = pd.concat([df1, df2, df3, df4])
        df2["score"] = eval_dist
        title_score = []  # in_so, so, in_ta, ta, official
        official_score = []
        y_pred = df2.loc[:, "score"].to_numpy()
        y_true = 1 - df2.loc[:, "is_normal"].to_numpy()
        pauc = roc_auc_score(y_true, y_pred, max_fpr=0.1) * 100
        official_score.append(pauc)
        for domain in ["source", "target"]:
            idx = df2["domain"] == domain
            y_pred = df2.loc[idx, "score"].to_numpy()
            y_true = 1 - df2.loc[idx, "is_normal"].to_numpy()
            auc = roc_auc_score(y_true, y_pred) * 100
            title_score.append(auc)
            idx = idx | ~df["is_normal"]
            y_pred = df2.loc[idx, "score"].to_numpy()
            y_true = 1 - df2.loc[idx, "is_normal"].to_numpy()
            auc = roc_auc_score(y_true, y_pred) * 100
            official_score.append(auc)
            title_score.append(auc)
        title_score.append(hmean(official_score))
        save_path = f"{tag}/epoch{epoch}/umap_{machine}_epoch{epoch}.png"

        so_ce_atts = [att for att in umap_df["att"].unique() if att.startswith("so_ce")]
        ta_ce_atts = [att for att in umap_df["att"].unique() if att.startswith("ta_ce")]
        non_ce_atts = [
            att for att in umap_df["att"].unique() if not att.startswith("ce")
        ]

        so_ce_palette = sns.color_palette(
            "coolwarm", len(so_ce_atts)
        )  
        ta_ce_palette = sns.color_palette("husl", len(ta_ce_atts))
        non_ce_palette = sns.color_palette(
            "Paired", len(non_ce_atts)
        )  
        color_dict = dict(zip(so_ce_atts, so_ce_palette))
        color_dict.update(dict(zip(ta_ce_atts, ta_ce_palette)))
        color_dict.update(dict(zip(non_ce_atts, non_ce_palette)))
        plt.figure(figsize=(10, 6))
        for att in sorted(umap_df["att"].unique()):
            subset = umap_df[(umap_df["att"] == att) & (umap_df["is_normal"])]
            for _, row in subset.iterrows():
                if row["att"].startswith("so_ce"):
                    plt.scatter(
                        row["u0"],
                        row["u1"],
                        edgecolors="black",
                        marker="^",
                        label=(
                            ""
                            if att not in plt.gca().get_legend_handles_labels()[1]
                            else ""
                        ),
                        s=80,
                        facecolors="none",
                    )
                elif row["att"].startswith("ta_ce"):
                    plt.scatter(
                        row["u0"],
                        row["u1"],
                        edgecolors="black",
                        marker="s",
                        label=(
                            ""
                            if att not in plt.gca().get_legend_handles_labels()[1]
                            else ""
                        ),
                        s=80,
                        facecolors="none",
                    )
                else:
                    plt.scatter(
                        row["u0"],
                        row["u1"],
                        edgecolors=color_dict[row["att"]],
                        marker="o",
                        label=(
                            row["att"]
                            if att not in plt.gca().get_legend_handles_labels()[1]
                            else ""
                        ),
                        s=40,
                        facecolors="none",
                    )

        for att in sorted(umap_df["att"].unique()):
            subset = umap_df[(umap_df["att"] == att) & (~umap_df["is_normal"])]
            for _, row in subset.iterrows():
                plt.scatter(
                    row["u0"],
                    row["u1"],
                    color=color_dict[row["att"]],
                    marker="x",
                    label=(
                        row["att"]
                        if att not in plt.gca().get_legend_handles_labels()[1]
                        else ""
                    ),
                    s=80,
                )

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title(
            f"Epoch{epoch} {machine}: in_so:{title_score[0]:.1f} so:{title_score[1]:.1f} in_ta:{title_score[2]:.1f} ta:{title_score[3]:.1f} offical:{title_score[4]:.1f}"
        )
        plt.tight_layout()
        plt.savefig(save_path)
        logging.info(save_path)
        train_cos = np.min(
            1 - np.dot(x_train_ln, means_target_ln.transpose()), axis=-1, keepdims=True
        )
        train_cos = np.minimum(
            train_cos,
            np.min(
                1 - np.dot(x_train_ln, means_source_ln.transpose()),
                axis=-1,
                keepdims=True,
            ),
        )
        centroids[lab, :n_so_centroids] = means_source_ln
        centroids[lab, -all_target:] = means_target_ln

        # assign cluster
        x_lab = x_eval_ln
        ta_distances = ta_kmeans.transform(x_lab)
        so_distances = so_kmeans.transform(x_lab)
        ta_pred_clusters = np.argmin(ta_distances, axis=1)
        so_pred_clusters = np.argmin(so_distances, axis=1)
        ta_pred_clusters += n_so_centroids
        machine_pred_cluster = []
        for ta_dist, so_dist, ta_cluster, so_cluster in zip(
            np.min(ta_distances, axis=1),
            np.min(so_distances, axis=1),
            ta_pred_clusters,
            so_pred_clusters,
        ):
            if ta_dist < so_dist:
                machine_pred_cluster.append(ta_cluster)
            else:
                machine_pred_cluster.append(so_cluster)
        epoch_eval_df[columns[5]] = eval_dist
        epoch_eval_df[columns[6]] = np.array(machine_pred_cluster)
        epoch_eval_df_list.append(epoch_eval_df)

        x_lab = x_train_ln
        ta_distances = ta_kmeans.transform(x_lab)
        so_distances = so_kmeans.transform(x_lab)
        ta_pred_clusters = np.argmin(ta_distances, axis=1)
        so_pred_clusters = np.argmin(so_distances, axis=1)
        ta_pred_clusters += n_so_centroids
        machine_pred_cluster = []
        for ta_dist, so_dist, ta_cluster, so_cluster in zip(
            np.min(ta_distances, axis=1),
            np.min(so_distances, axis=1),
            ta_pred_clusters,
            so_pred_clusters,
        ):
            if ta_dist < so_dist:
                machine_pred_cluster.append(ta_cluster)
            else:
                machine_pred_cluster.append(so_cluster)
        epoch_train_df[columns[5]] = np.min(train_cos, axis=-1)
        epoch_train_df[columns[6]] = np.array(machine_pred_cluster)
        epoch_train_df_list.append(epoch_train_df)
    epoch_eval_df = pd.concat(epoch_eval_df_list)
    epoch_train_df = pd.concat(epoch_train_df_list)
    epoch_train_df[columns].to_csv(f"{model_dir}/train_{kmeans_mode}.csv", index=False)
    epoch_eval_df[columns].to_csv(f"{model_dir}/eval_{kmeans_mode}.csv", index=False)
    score_df = get_score_df(epoch_eval_df) * 100
    score_df.to_csv(f"{model_dir}/score_{kmeans_mode}.csv")
    save_path = f"{tag}/epoch{epoch}/kmeans_so{n_so_centroids}_ta{all_target}_epoch{epoch}_seed{seed}.png"
    logging.info(save_path)
