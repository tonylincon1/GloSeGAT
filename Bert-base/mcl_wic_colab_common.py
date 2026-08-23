import argparse
import gc
import json
import math
import os
import random
import subprocess
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ModuleNotFoundError:
    nltk = None
    wn = None


DEFAULT_MODEL_NAME = "bert-base-multilingual-cased"
DEFAULT_DATA_ROOT = "/content/mcl-wic/extracted"
TARGET_START = "[TGT]"
TARGET_END = "[/TGT]"
POS_LABELS = ["NOUN", "VERB", "ADJ", "ADV", "PROPN", "OTHER"]
MCL_TO_OMW_LANG = {
    "en": "eng",
    "fr": "fra",
    "ru": "rus",
    "zh": "cmn",
    "ar": "arb",
}
ALL_TEST_SPLITS = [
    "test.ar-ar",
    "test.en-en",
    "test.fr-fr",
    "test.ru-ru",
    "test.zh-zh",
    "test.en-ar",
    "test.en-fr",
    "test.en-ru",
    "test.en-zh",
]


@dataclass
class ExperimentConfig:
    variant: str
    data_root: str
    output_dir: str
    model_name: str
    n_trials: int
    n_seed_runs: int
    max_epochs: int
    patience: int
    max_len: int
    seed: int
    seed_start: int
    train_limit: Optional[int]
    eval_limit: Optional[int]
    num_workers: int
    device: str
    freeze_encoder: bool
    download_data: bool


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def clear_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def ensure_wordnet() -> None:
    global nltk, wn
    if nltk is None or wn is None:
        import nltk as nltk_module
        from nltk.corpus import wordnet as wn_module
        nltk = nltk_module
        wn = wn_module
    try:
        _ = wn.synsets("dog")
        _ = wn.synsets("chien", lang="fra")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")


def ensure_mcl_wic_data(data_root: Path, download_data: bool = True) -> None:
    expected = data_root / "MCL-WiC" / "training" / "training.en-en.data"
    expected_gold = data_root / "test.en-en.gold"
    if expected.exists() and expected_gold.exists():
        return
    if not download_data:
        raise FileNotFoundError(f"MCL-WiC data not found at {data_root}")

    work_dir = data_root.parent
    repo_dir = work_dir / "mcl-wic"
    work_dir.mkdir(parents=True, exist_ok=True)
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/SapienzaNLP/mcl-wic.git", str(repo_dir)],
            check=True,
        )

    data_root.mkdir(parents=True, exist_ok=True)
    for zip_name in [
        "SemEval-2021_MCL-WiC_all-datasets.zip",
        "SemEval-2021_MCL-WiC_test-gold-data.zip",
    ]:
        with zipfile.ZipFile(repo_dir / zip_name) as zf:
            zf.extractall(data_root)


def load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_lang_pair(split_name: str) -> Tuple[str, str]:
    pair = split_name.split(".", 1)[1]
    lang1, lang2 = pair.split("-")
    return lang1, lang2


def load_split(data_path: Path, gold_path: Path, split_name: str, limit: Optional[int] = None) -> List[dict]:
    rows = load_json(data_path)
    gold_rows = load_json(gold_path)
    gold_by_id = {row["id"]: row["tag"] for row in gold_rows}
    lang1, lang2 = split_lang_pair(split_name)
    merged = []
    for row in rows:
        if row["id"] not in gold_by_id:
            continue
        item = dict(row)
        item["tag"] = gold_by_id[row["id"]]
        item["label"] = 1 if item["tag"] == "T" else 0
        item["split_name"] = split_name
        item["lang_pair"] = f"{lang1}-{lang2}"
        item["lang1"] = lang1
        item["lang2"] = lang2
        # MCL-WiC gives one lemma; for cross-lingual files it is the first-side
        # lemma in the official data, so lang1 is the safest OMW lookup language.
        item["wn_lang"] = MCL_TO_OMW_LANG.get(lang1, "eng")
        merged.append(item)
    return merged[:limit] if limit else merged


def split_paths(data_root: Path) -> Dict[str, Tuple[Path, Path]]:
    base = data_root / "MCL-WiC"
    paths = {
        "train.en-en": (
            base / "training" / "training.en-en.data",
            base / "training" / "training.en-en.gold",
        ),
        "dev.en-en": (
            base / "dev" / "multilingual" / "dev.en-en.data",
            base / "dev" / "multilingual" / "dev.en-en.gold",
        ),
    }
    for pair in ["ar-ar", "en-en", "fr-fr", "ru-ru", "zh-zh"]:
        paths[f"test.{pair}"] = (
            base / "test" / "multilingual" / f"test.{pair}.data",
            data_root / f"test.{pair}.gold",
        )
    for pair in ["en-ar", "en-fr", "en-ru", "en-zh"]:
        paths[f"test.{pair}"] = (
            base / "test" / "crosslingual" / f"test.{pair}.data",
            data_root / f"test.{pair}.gold",
        )
    return paths


def load_all_rows(cfg: ExperimentConfig) -> Dict[str, List[dict]]:
    data_root = Path(cfg.data_root)
    ensure_mcl_wic_data(data_root, cfg.download_data)
    rows_by_split = {}
    for split_name, (data_path, gold_path) in split_paths(data_root).items():
        limit = cfg.train_limit if split_name == "train.en-en" else cfg.eval_limit
        rows_by_split[split_name] = load_split(data_path, gold_path, split_name, limit)
    return rows_by_split


def first_span(row: dict, side: int) -> Tuple[int, int]:
    if f"start{side}" in row and f"end{side}" in row:
        return int(row[f"start{side}"]), int(row[f"end{side}"])
    span = str(row[f"ranges{side}"]).split(",")[0].strip()
    start, end = span.split("-")
    return int(start), int(end)


def insert_markers(sentence: str, start: int, end: int) -> str:
    return f"{sentence[:start]} {TARGET_START} {sentence[start:end]} {TARGET_END} {sentence[end:]}"


def pos_id(pos: str) -> int:
    pos = (pos or "OTHER").upper()
    return POS_LABELS.index(pos) if pos in POS_LABELS else POS_LABELS.index("OTHER")


def wicpos_to_wn(pos: str):
    ensure_wordnet()
    pos = (pos or "").upper()
    if pos in {"N", "NOUN", "PROPN"}:
        return wn.NOUN
    if pos in {"V", "VERB"}:
        return wn.VERB
    if pos in {"A", "J", "ADJ"}:
        return wn.ADJ
    if pos in {"R", "ADV"}:
        return wn.ADV
    return None


def gloss_text(synset) -> str:
    examples = " ".join(synset.examples()[:1])
    return f"{synset.name().split('.')[0]} ({synset.pos()}): {synset.definition()}. {examples}".strip()


def synset_neighbors(synset) -> set:
    neighbors = set()
    for rel in [synset.hypernyms(), synset.hyponyms(), synset.similar_tos(), synset.also_sees()]:
        for node in rel:
            neighbors.add(node.name())
    return neighbors


def build_candidate_adj(candidate_synsets: Sequence) -> torch.Tensor:
    size = len(candidate_synsets)
    if size == 0:
        return torch.zeros((0, 0), dtype=torch.float)

    names = [synset.name() for synset in candidate_synsets]
    lexnames = [synset.lexname() for synset in candidate_synsets]
    neighbor_maps = [synset_neighbors(synset) for synset in candidate_synsets]
    adj = torch.eye(size, dtype=torch.float)

    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            linked = names[j] in neighbor_maps[i] or names[i] in neighbor_maps[j]
            same_lexname = lexnames[i] == lexnames[j]
            if linked or same_lexname:
                adj[i, j] = 1.0
    return adj


class PairDataset(Dataset):
    def __init__(self, rows: Sequence[dict], tokenizer, max_len: int, mark_targets: bool):
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mark_targets = mark_targets

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        s1, s2 = row["sentence1"], row["sentence2"]
        if self.mark_targets:
            s1 = insert_markers(s1, *first_span(row, 1))
            s2 = insert_markers(s2, *first_span(row, 2))
        enc = self.tokenizer(
            s1,
            s2,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(row["label"], dtype=torch.float)
        return item


class TargetDataset(Dataset):
    def __init__(self, rows: Sequence[dict], tokenizer, max_len: int):
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def _encode_sentence(self, sentence: str, span: Tuple[int, int]) -> Tuple[dict, int]:
        enc = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True,
        )
        target_index = 1
        start, end = span
        for i, (a, b) in enumerate(enc["offset_mapping"]):
            if a == b:
                continue
            if not (b <= start or a >= end):
                target_index = i
                break
        enc.pop("offset_mapping", None)
        return enc, target_index

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        enc1, target1 = self._encode_sentence(row["sentence1"], first_span(row, 1))
        enc2, target2 = self._encode_sentence(row["sentence2"], first_span(row, 2))
        lemma_enc = self.tokenizer(
            row["lemma"],
            truncation=True,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids_1": torch.tensor(enc1["input_ids"], dtype=torch.long),
            "attention_mask_1": torch.tensor(enc1["attention_mask"], dtype=torch.long),
            "target_index_1": torch.tensor(target1, dtype=torch.long),
            "input_ids_2": torch.tensor(enc2["input_ids"], dtype=torch.long),
            "attention_mask_2": torch.tensor(enc2["attention_mask"], dtype=torch.long),
            "target_index_2": torch.tensor(target2, dtype=torch.long),
            "lemma_input_ids": lemma_enc["input_ids"].squeeze(0),
            "lemma_attention_mask": lemma_enc["attention_mask"].squeeze(0),
            "pos_ids": torch.tensor(pos_id(row.get("pos", "OTHER")), dtype=torch.long),
            "labels": torch.tensor(row["label"], dtype=torch.float),
            "lemmas": row["lemma"],
            "poses": row.get("pos", "OTHER"),
            "wn_langs": row.get("wn_lang", "eng"),
        }


def pad_target_batch(batch: List[dict], pad_id: int) -> dict:
    def pad_key(key: str, pad_value: int) -> torch.Tensor:
        seqs = [item[key] for item in batch]
        max_size = max(seq.size(0) for seq in seqs)
        out = []
        for seq in seqs:
            pad = max_size - seq.size(0)
            if pad:
                seq = torch.cat([seq, torch.full((pad,), pad_value, dtype=seq.dtype)])
            out.append(seq)
        return torch.stack(out)

    return {
        "input_ids_1": pad_key("input_ids_1", pad_id),
        "attention_mask_1": pad_key("attention_mask_1", 0),
        "target_index_1": torch.stack([item["target_index_1"] for item in batch]),
        "input_ids_2": pad_key("input_ids_2", pad_id),
        "attention_mask_2": pad_key("attention_mask_2", 0),
        "target_index_2": torch.stack([item["target_index_2"] for item in batch]),
        "lemma_input_ids": torch.stack([item["lemma_input_ids"] for item in batch]),
        "lemma_attention_mask": torch.stack([item["lemma_attention_mask"] for item in batch]),
        "pos_ids": torch.stack([item["pos_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "lemmas": [item["lemmas"] for item in batch],
        "poses": [item["poses"] for item in batch],
        "wn_langs": [item["wn_langs"] for item in batch],
    }


def is_encoder_frozen(encoder: nn.Module) -> bool:
    return not any(param.requires_grad for param in encoder.parameters())


class PairBertClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        kwargs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"]
        out = self.encoder(**kwargs)
        pooled = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled)).squeeze(-1)


class TargetFeatureClassifier(nn.Module):
    def __init__(self, model_name: str, use_pos: bool, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.use_pos = use_pos
        self.pos_embedding = nn.Embedding(len(POS_LABELS), hidden) if use_pos else None
        input_dim = hidden * 4 + 1 + (hidden if use_pos else 0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def pool_target(self, last_hidden: torch.Tensor, target_index: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden = last_hidden.shape
        index = target_index.clamp(0, seq_len - 1).view(batch_size, 1, 1).expand(batch_size, 1, hidden)
        return last_hidden.gather(1, index).squeeze(1)

    def encode_target(self, input_ids, attention_mask, target_index):
        if is_encoder_frozen(self.encoder):
            with torch.no_grad():
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.pool_target(out.last_hidden_state, target_index)

    def forward(self, batch: dict) -> torch.Tensor:
        h1 = self.encode_target(batch["input_ids_1"], batch["attention_mask_1"], batch["target_index_1"])
        h2 = self.encode_target(batch["input_ids_2"], batch["attention_mask_2"], batch["target_index_2"])
        cos = F.cosine_similarity(h1, h2, dim=-1).unsqueeze(-1)
        features = [h1, h2, (h1 - h2).abs(), h1 * h2, cos]
        if self.use_pos:
            features.append(self.pos_embedding(batch["pos_ids"]))
        return self.classifier(torch.cat(features, dim=-1)).squeeze(-1)


class DenseGATLayer(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_src = nn.Linear(hidden_size, 1, bias=False)
        self.attn_dst = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        projected = self.proj(nodes)
        scores = self.attn_src(projected) + self.attn_dst(projected).transpose(1, 2)
        alpha = torch.softmax(F.leaky_relu(scores, negative_slope=0.2), dim=-1)
        updated = alpha @ projected
        return self.norm(nodes + self.dropout(F.elu(updated)))


class SenseCache:
    def __init__(self, topk: int):
        self.topk = topk
        self.meta = {}
        self.vecs = {}

    def reset_vecs(self) -> None:
        self.vecs = {}

    def get_synsets(self, lemma: str, pos: str, wn_lang: str) -> Dict:
        key = (str(lemma).lower(), str(pos).upper(), wn_lang, self.topk)
        if key in self.meta:
            return self.meta[key]

        wn_pos = wicpos_to_wn(pos)
        synsets = []
        for lang in [wn_lang, "eng"]:
            try:
                candidates = wn.synsets(lemma, pos=wn_pos, lang=lang) if wn_pos else wn.synsets(lemma, lang=lang)
            except Exception:
                candidates = []
            if candidates:
                synsets = candidates
                break

        seen = set()
        unique = []
        for synset in synsets:
            if synset.name() in seen:
                continue
            unique.append(synset)
            seen.add(synset.name())
            if len(unique) >= self.topk:
                break

        self.meta[key] = {
            "synsets": unique,
            "names": [synset.name() for synset in unique],
            "glosses": [gloss_text(synset) for synset in unique],
            "lookup_lang": wn_lang,
        }
        return self.meta[key]

    def get_gloss_embeddings(self, lemma: str, pos: str, wn_lang: str, encoder, tokenizer, device, max_len: int):
        meta = self.get_synsets(lemma, pos, wn_lang)
        cache_key = (str(lemma).lower(), str(pos).upper(), wn_lang, device.type, self.topk, max_len)
        if cache_key in self.vecs:
            return meta, self.vecs[cache_key]
        if not meta["glosses"]:
            emb = torch.zeros((0, encoder.config.hidden_size), device=device)
            self.vecs[cache_key] = emb
            return meta, emb

        enc = tokenizer(
            meta["glosses"],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = encoder(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        self.vecs[cache_key] = emb.detach()
        return meta, self.vecs[cache_key]


class SmallSenseGAT(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn = nn.Linear(hidden_size * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if nodes.size(0) == 0:
            return nodes
        projected = self.proj(nodes)
        size, hidden = projected.shape
        src = projected.unsqueeze(1).expand(size, size, hidden)
        dst = projected.unsqueeze(0).expand(size, size, hidden)
        scores = F.leaky_relu(self.attn(torch.cat([src, dst], dim=-1)).squeeze(-1), negative_slope=0.2)
        scores = scores.masked_fill(adj <= 0, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        updated = alpha @ projected
        return self.norm(nodes + self.dropout(F.elu(updated)))


class TargetGATClassifier(TargetFeatureClassifier):
    """
    Public variant name: glowic_gat.

    Internally this follows GLOSS_PLUS_GRAPH: candidate synsets are retrieved
    from WordNet/Open Multilingual WordNet, represented by English gloss
    embeddings, used both as raw gloss features and as graph-refined features
    after a GAT over WordNet relations.
    """
    def __init__(self, model_name: str, dropout: float, topk: int, max_len: int):
        super().__init__(model_name=model_name, use_pos=False, dropout=dropout)
        hidden = self.encoder.config.hidden_size
        self.topk = topk
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.sense_cache = SenseCache(topk=topk)
        self.sense_gat = SmallSenseGAT(hidden, dropout)
        self.cos = nn.CosineSimilarity(dim=-1)
        input_dim = hidden * 12 + 4
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def reset_sense_cache(self) -> None:
        self.sense_cache.reset_vecs()

    def attend_candidates(self, h: torch.Tensor, candidate_emb: torch.Tensor) -> torch.Tensor:
        if candidate_emb.size(0) == 0:
            return torch.zeros_like(h)
        sim = F.cosine_similarity(h.unsqueeze(1), candidate_emb.unsqueeze(0), dim=-1)
        weights = F.softmax(sim, dim=-1)
        return weights @ candidate_emb

    def get_candidate_embeddings(self, lemma: str, pos: str, wn_lang: str, device: torch.device):
        meta, gloss_emb = self.sense_cache.get_gloss_embeddings(
            lemma,
            pos,
            wn_lang,
            self.encoder,
            self.tokenizer,
            device,
            self.max_len,
        )
        if gloss_emb.size(0) == 0:
            return gloss_emb, gloss_emb
        adj = build_candidate_adj(meta["synsets"]).to(device)
        graph_emb = self.sense_gat(gloss_emb, adj)
        return gloss_emb, graph_emb

    def forward(self, batch: dict) -> torch.Tensor:
        device = batch["input_ids_1"].device
        h1 = self.encode_target(batch["input_ids_1"], batch["attention_mask_1"], batch["target_index_1"])
        h2 = self.encode_target(batch["input_ids_2"], batch["attention_mask_2"], batch["target_index_2"])

        z1_gloss_list, z2_gloss_list = [], []
        z1_graph_list, z2_graph_list = [], []
        for b, (lemma, pos, wn_lang) in enumerate(zip(batch["lemmas"], batch["poses"], batch["wn_langs"])):
            gloss_emb, graph_emb = self.get_candidate_embeddings(lemma, pos, wn_lang, device)
            z1_gloss_list.append(self.attend_candidates(h1[b:b + 1], gloss_emb))
            z2_gloss_list.append(self.attend_candidates(h2[b:b + 1], gloss_emb))
            z1_graph_list.append(self.attend_candidates(h1[b:b + 1], graph_emb))
            z2_graph_list.append(self.attend_candidates(h2[b:b + 1], graph_emb))

        z1_gloss = torch.cat(z1_gloss_list, dim=0)
        z2_gloss = torch.cat(z2_gloss_list, dim=0)
        z1_graph = torch.cat(z1_graph_list, dim=0)
        z2_graph = torch.cat(z2_graph_list, dim=0)

        features = torch.cat(
            [
                h1,
                h2,
                (h1 - h2).abs(),
                h1 * h2,
                z1_gloss,
                z2_gloss,
                (z1_gloss - z2_gloss).abs(),
                z1_gloss * z2_gloss,
                self.cos(h1, z1_gloss).unsqueeze(-1),
                self.cos(h2, z2_gloss).unsqueeze(-1),
                z1_graph,
                z2_graph,
                (z1_graph - z2_graph).abs(),
                z1_graph * z2_graph,
                self.cos(h1, z1_graph).unsqueeze(-1),
                self.cos(h2, z2_graph).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.classifier(features).squeeze(-1)


def build_model(variant: str, model_name: str, dropout: float, topk: int = 5, max_len: int = 128) -> nn.Module:
    if variant in {"bert_cls", "bert_marked"}:
        return PairBertClassifier(model_name, dropout)
    if variant == "glowic_target":
        return TargetFeatureClassifier(model_name, use_pos=False, dropout=dropout)
    if variant == "glowic_target_pos":
        return TargetFeatureClassifier(model_name, use_pos=True, dropout=dropout)
    if variant == "glowic_gat":
        return TargetGATClassifier(model_name, dropout, topk=topk, max_len=max_len)
    raise ValueError(f"Unknown variant: {variant}")


def build_datasets(variant: str, rows_by_split: Dict[str, List[dict]], tokenizer, max_len: int):
    if variant == "bert_cls":
        return {k: PairDataset(v, tokenizer, max_len, mark_targets=False) for k, v in rows_by_split.items()}, False
    if variant == "bert_marked":
        return {k: PairDataset(v, tokenizer, max_len, mark_targets=True) for k, v in rows_by_split.items()}, False
    return {k: TargetDataset(v, tokenizer, max_len) for k, v in rows_by_split.items()}, True


def make_loaders(datasets, tokenizer, batch_size: int, num_workers: int, target_style: bool, device: torch.device):
    collate_fn = (lambda batch: pad_target_batch(batch, tokenizer.pad_token_id or 0)) if target_style else None
    return {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train.en-en",
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
        )
        for name, dataset in datasets.items()
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def metrics_from_logits(labels: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch)
        labels = batch["labels"]
        total_loss += F.binary_cross_entropy_with_logits(logits, labels).item()
        all_logits.append(logits.detach().cpu().float().numpy())
        all_labels.append(labels.detach().cpu().long().numpy())
    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)
    metrics = metrics_from_logits(labels_np, logits_np)
    metrics["loss"] = float(total_loss / max(len(loader), 1))
    return metrics


def optimizer_for(model: nn.Module, lr_encoder: float, lr_head: float, weight_decay: float):
    encoder_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": lr_encoder, "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
    return torch.optim.AdamW(groups)


def suggest_hparams(trial: optuna.Trial, variant: str) -> Dict:
    pair_variant = variant in {"bert_cls", "bert_marked"}
    batch_choices = [8, 16] if pair_variant else [2, 4, 8]
    hparams = {
        "lr_encoder": trial.suggest_float("lr_encoder", 1e-5, 5e-5, log=True),
        "lr_head": trial.suggest_float("lr_head", 5e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "dropout": trial.suggest_float("dropout", 0.05, 0.35),
        "batch_size": trial.suggest_categorical("batch_size", batch_choices),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.15),
    }
    if variant == "glowic_gat":
        hparams["topk"] = trial.suggest_int("topk", 1, 8)
    return hparams


def train_once(
    cfg: ExperimentConfig,
    rows_by_split: Dict[str, List[dict]],
    tokenizer,
    hparams: Dict,
    seed: int,
    eval_splits: List[str],
) -> Dict:
    clear_device_cache()
    set_seed(seed)
    device = pick_device(cfg.device)
    datasets, target_style = build_datasets(cfg.variant, rows_by_split, tokenizer, cfg.max_len)
    loaders = make_loaders(datasets, tokenizer, hparams["batch_size"], cfg.num_workers, target_style, device)
    model = build_model(
        cfg.variant,
        cfg.model_name,
        hparams["dropout"],
        topk=int(hparams.get("topk", 5)),
        max_len=cfg.max_len,
    ).to(device)
    model.encoder.resize_token_embeddings(len(tokenizer))
    if cfg.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    optimizer = optimizer_for(model, hparams["lr_encoder"], hparams["lr_head"], hparams["weight_decay"])
    total_steps = cfg.max_epochs * math.ceil(len(loaders["train.en-en"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * hparams["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    best_dev_accuracy = -1.0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        if hasattr(model, "reset_sense_cache"):
            model.reset_sense_cache()
        total_train_loss = 0.0
        pbar = tqdm(loaders["train.en-en"], desc=f"{cfg.variant} seed={seed} epoch={epoch}", leave=False)
        for batch in pbar:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        if hasattr(model, "reset_sense_cache"):
            model.reset_sense_cache()
        dev_metrics = evaluate(model, loaders["dev.en-en"], device)
        epoch_log = {
            "epoch": epoch,
            "train_loop_loss": float(total_train_loss / max(len(loaders["train.en-en"]), 1)),
            "dev": dev_metrics,
        }
        history.append(epoch_log)
        print(f"[{cfg.variant}] seed={seed} epoch={epoch} dev_acc={dev_metrics['accuracy']:.4f}")

        if dev_metrics["accuracy"] > best_dev_accuracy:
            best_dev_accuracy = dev_metrics["accuracy"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if hasattr(model, "reset_sense_cache"):
        model.reset_sense_cache()
    final = {split: evaluate(model, loaders[split], device) for split in eval_splits}
    result = {
        "seed": seed,
        "hparams": hparams,
        "best_dev_accuracy": float(best_dev_accuracy),
        "history": history,
        "final": final,
    }
    del model, optimizer, scheduler, loaders, datasets
    clear_device_cache()
    return result


def aggregate_seed_runs(seed_runs: List[Dict]) -> Dict:
    splits = sorted(seed_runs[0]["final"].keys()) if seed_runs else []
    aggregate = {}
    for split in splits:
        aggregate[split] = {}
        for metric in ["accuracy", "precision", "recall", "f1", "loss"]:
            values = [run["final"][split][metric] for run in seed_runs]
            aggregate[split][f"{metric}_mean"] = float(np.mean(values))
            aggregate[split][f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return aggregate


def run_colab_experiment(variant: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default=f"/content/mcl_wic_results/{variant}")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-seed-runs", type=int, default=15)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--no-download-data", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = ExperimentConfig(
        variant=variant,
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        n_trials=args.n_trials,
        n_seed_runs=args.n_seed_runs,
        max_epochs=args.max_epochs,
        patience=args.patience,
        max_len=args.max_len,
        seed=args.seed,
        seed_start=args.seed_start,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        num_workers=args.num_workers,
        device=str(pick_device(args.device)),
        freeze_encoder=args.freeze_encoder,
        download_data=not args.no_download_data,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.variant == "glowic_gat":
        ensure_wordnet()
    rows_by_split = load_all_rows(cfg)
    print("Config:", asdict(cfg))
    print("Dataset sizes:", {k: len(v) for k, v in rows_by_split.items()})

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [TARGET_START, TARGET_END]})

    def objective(trial: optuna.Trial) -> float:
        hparams = suggest_hparams(trial, cfg.variant)
        try:
            result = train_once(
                cfg,
                rows_by_split,
                tokenizer,
                hparams,
                seed=cfg.seed,
                eval_splits=["dev.en-en"],
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                clear_device_cache()
                raise optuna.TrialPruned(str(exc))
            raise
        trial.set_user_attr("result", result)
        trial.set_user_attr("hparams", hparams)
        return result["best_dev_accuracy"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    study.optimize(objective, n_trials=cfg.n_trials, gc_after_trial=True)

    best_hparams = dict(study.best_trial.user_attrs["hparams"])
    trials_export = []
    for trial in study.trials:
        trials_export.append(
            {
                "number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
        )
    with (output_dir / "optuna_trials.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "best_trial_number": study.best_trial.number,
                "best_dev_accuracy": study.best_value,
                "best_hparams": best_hparams,
                "trials": trials_export,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    seed_runs = []
    eval_splits = ["dev.en-en"] + ALL_TEST_SPLITS
    for i in range(cfg.n_seed_runs):
        seed = cfg.seed_start + i
        run = train_once(cfg, rows_by_split, tokenizer, best_hparams, seed=seed, eval_splits=eval_splits)
        seed_runs.append(run)
        with (output_dir / f"seed_{seed}.json").open("w", encoding="utf-8") as f:
            json.dump({"config": asdict(cfg), "best_hparams": best_hparams, "run": run}, f, indent=2, ensure_ascii=False)

    summary = {
        "config": asdict(cfg),
        "best_hparams": best_hparams,
        "best_optuna_dev_accuracy": study.best_value,
        "seed_runs": seed_runs,
        "aggregate": aggregate_seed_runs(seed_runs),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {output_dir}")
