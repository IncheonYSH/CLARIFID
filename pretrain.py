import os

import math
from typing import List, Dict, Any, Optional
import numpy as np
import time
import random
from datetime import timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
import timm
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader

from transformers import (
    GPT2Config, GPT2Tokenizer,
    BertConfig, BertModel, BertTokenizer,
    GPT2LMHeadModel, GPT2Model
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import evaluate

meteor = evaluate.load("meteor")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
from datasets import Dataset as HFDataset

import pandas as pd
from PIL import Image
import logging
import sys
from contextlib import nullcontext
import re


#
# ----------- Cosine LR Decay Helper (only once per "outer iteration") ----------
#
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    if it > lr_decay_iters:
        return min_lr
    if it < warmup_iters:
        return learning_rate * float(it + 1) / float(warmup_iters + 1)
    decay_ratio = (it - warmup_iters) / float(lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def setup_logger(log_dir: Optional[str] = None):
    timestamp = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        logfile = os.path.join(log_dir, f"train_log_{timestamp}.log")
    else:
        logfile = f"train_log_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()
    logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        fmt="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)

    file_handler = logging.FileHandler(logfile, mode="w")
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        fmt="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger, logfile


def setup_distributed(args):
    ddp = int(os.environ.get('RANK', -1)) != -1

    if ddp:
        init_process_group(
            backend=args.backend,
            timeout=timedelta(hours=5)
        )
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])

        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)
    else:
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return ddp, device, master_process, ddp_world_size


def cleanup_distributed():
    if torch.distributed.is_initialized():
        destroy_process_group()

class MIMICCXRIMAGEDataset(Dataset):
    def __init__(self, csv_file, transform=None, split_mode="train", filter_findings=True, max_images=3):
        df = pd.read_csv(csv_file)
        if split_mode == "train":
            df = df[df["split"].isin(["train", "validate"])]
        elif split_mode == "test":
            df = df[df["split"] == "test"]
        else:
            raise ValueError("split_mode must be 'train' or 'test'")
        # df = df[(df["has_impression"] == 1)]
        #
        # df = df[
        #     (df["impression_tokens_gpt2"] >= 4)
        #     ]
        # # Reset index after filtering
        # df = df.reset_index(drop=True)

        valid_rows = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            image_paths = row["image_paths"].split(";")
            if len(image_paths) <= max_images:
                valid_rows.append(row)
        if len(valid_rows) > 0:
            df = pd.DataFrame(valid_rows)
        else:
            df = pd.DataFrame(columns=df.columns)

        self.data = df.reset_index(drop=True)
        self.transform = transform
        logging.info(f"Dataset loaded with {len(self.data)} samples after filtering.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_paths = row["image_paths"].split(";")
        images = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        sample = {
            "subject_id": row["subject_id"],
            "study_id": row["study_id"],
            "split": row["split"],
            "images": images,
            "findings": row["findings"],
            "impression": row["impression"],
            "last_paragraph": row["last_paragraph"],
            "report_all": row["report_all"],
            "chexpert_labels": [int(x) for x in row["chexpert_labels"].split(";")],
            "chexbert_labels": [int(x) for x in row["chexbert_labels"].split(";")]
        }
        return sample

class MIMICCXRDataset(Dataset):
    def __init__(self, csv_file, transform=None, split_mode="train", filter_findings=True, max_images=3):
        df = pd.read_csv(csv_file)
        if split_mode == "train":
            df = df[df["split"].isin(["train", "validate"])]
        elif split_mode == "test":
            df = df[df["split"] == "test"]
        else:
            raise ValueError("split_mode must be 'train' or 'test'")

        if filter_findings:
            if filter_findings:
                df = df[(df["has_impression"] == 1) & (df["has_findings"] == 1)]

                # Additional filters:
                # 1) findings_tokens_gpt2 > impression_tokens_gpt2
                # 2) impression_tokens_gpt2 >= 5
                df = df[
                    (df["findings_tokens_gpt2"] > df["impression_tokens_gpt2"]) &
                    (df["impression_tokens_gpt2"] >= 4)
                    ]

                # Reset index after filtering
                df = df.reset_index(drop=True)

        valid_rows = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            image_paths = row["image_paths"].split(";")
            if len(image_paths) <= max_images:
                valid_rows.append(row)
        if len(valid_rows) > 0:
            df = pd.DataFrame(valid_rows)
        else:
            df = pd.DataFrame(columns=df.columns)

        self.data = df.reset_index(drop=True)
        self.transform = transform
        logging.info(f"Dataset loaded with {len(self.data)} samples after filtering.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_paths = row["image_paths"].split(";")
        images = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        sample = {
            "subject_id": row["subject_id"],
            "study_id": row["study_id"],
            "split": row["split"],
            "images": images,
            "findings": row["findings"],
            "impression": row["impression"],
            "last_paragraph": row["last_paragraph"],
            "report_all": row["report_all"],
            "chexpert_labels": [int(x) for x in row["chexpert_labels"].split(";")],
            "chexbert_labels": [int(x) for x in row["chexbert_labels"].split(";")]
        }
        return sample


def my_collate_fn(batch):
    out = {
        "subject_id": [],
        "study_id": [],
        "split": [],
        "images": [],
        "findings": [],
        "impression": [],
        "last_paragraph": [],
        "report_all": [],
        "chexpert_labels": [],
        "chexbert_labels": []
    }

    for sample in batch:
        out["subject_id"].append(sample["subject_id"])
        out["study_id"].append(sample["study_id"])
        out["split"].append(sample["split"])
        out["images"].append(sample["images"])
        out["findings"].append(sample["findings"])
        out["impression"].append(sample["impression"])
        out["last_paragraph"].append(sample["last_paragraph"])
        out["report_all"].append(sample["report_all"])
        out["chexpert_labels"].append(sample["chexpert_labels"])
        out["chexbert_labels"].append(sample["chexbert_labels"])

    return out


#
# 14 conditions + mapping
#
CONDITIONS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding"
]
CLASS_MAPPING = {
    0: "Blank",
    1: "Positive",
    2: "Negative",
    3: "Uncertain"
}


def build_training_reference(
        sample: dict,
        shuffle_findings: bool = True,
        use_report_tags: bool = True
) -> str:
    """
    Build ground-truth classification lines + <cls_end> + (Findings & Impression text).
    """
    labels_ = sample["chexbert_labels"]  # length=14
    cond_parts = []
    for idx_cond, cond_name in enumerate(CONDITIONS):
        cond_token = cond_name.replace(" ", "_")
        label_bin = labels_[idx_cond]
        class_str = "Positive" if label_bin == 1 else "Blank"
        part_i = f"<{cond_token}>:<{class_str}>"
        cond_parts.append(part_i)

    classification_block = ";".join(cond_parts)
    classification_block += " <cls_end>"

    def safe_str(x):
        if isinstance(x, str):
            return x.strip()
        if not x or isinstance(x, float):
            return ""
        return str(x).strip()

    findings_val = safe_str(sample.get("findings", ""))
    impression_val = safe_str(sample.get("impression", ""))

    find_sents = re.split(r'[.;]', findings_val) if findings_val else []
    find_sents = [fs.strip() for fs in find_sents if fs.strip()]
    if shuffle_findings and len(find_sents) > 1:
        random.shuffle(find_sents)

    chain = []
    if use_report_tags:
        chain.append("<findings>")
    chain.append("Findings:")
    if len(find_sents) == 0:
        if findings_val:
            chain.append(f"{findings_val}")
    else:
        for idx_, fsent in enumerate(find_sents, 1):
            chain.append(f"{fsent}.")
            if use_report_tags and idx_ < len(find_sents):
                chain.append("<next>")
    if impression_val:
        if use_report_tags:
            chain.append("<impression>")
        chain.append(f"Impression: {impression_val}")

    flexible_text = " ".join(chain)
    # final_ref = classification_block + " " + flexible_text
    final_ref = flexible_text
    return final_ref


meteor = evaluate.load("meteor")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


def _single_nlg_reward(ref: str, pred: str) -> dict:
    try:
        bleu_res = bleu.compute(predictions=[pred], references=[[ref]], max_order=1, smooth=True)
        bleu1_val = bleu_res["bleu"]
        bleu_res = bleu.compute(predictions=[pred], references=[[ref]], max_order=4, smooth=True)
        bleu4_val = bleu_res["bleu"]
    except:
        bleu1_val = 0
        bleu4_val = 0
    meteor_res = meteor.compute(predictions=[pred], references=[ref])
    meteor_val = float(meteor_res["meteor"])
    rouge_res  = rouge.compute(predictions=[pred], references=[ref], rouge_types=["rougeL"])
    rouge_val  = float(rouge_res["rougeL"])
    return {"BLEU_1": bleu1_val, "BLEU_4": bleu4_val, "METEOR": meteor_val, "ROUGE_L": rouge_val}





def compute_batch_nlg_metrics(ref_texts, list_of_preds, num_proc=4) -> dict:
    ds_dict = {"reference": ref_texts, "prediction": list_of_preds}
    ds = HFDataset.from_dict(ds_dict)

    def _map_fn(example):
        out = _single_nlg_reward(example["reference"], example["prediction"])
        return out

    ds_mapped = ds.map(_map_fn, batched=False, num_proc=num_proc)
    bleus_1 = ds_mapped["BLEU_1"]
    bleus_4 = ds_mapped["BLEU_4"]
    mets = ds_mapped["METEOR"]
    rous = ds_mapped["ROUGE_L"]
    n = len(bleus_4)
    if n > 0:
        return {
            "BLEU_1": float(np.mean(bleus_1)),
            "BLEU_4": float(np.mean(bleus_4)),
            "METEOR": float(np.mean(mets)),
            "ROUGE_L": float(np.mean(rous))
        }
    else:
        return {"BLEU_1": 0.0, "BLEU_4": 0.0, "METEOR": 0.0, "ROUGE_L": 0.0}


class CheXbert(nn.Module):
    def __init__(self, checkpoint_path, device, p=0.1):
        super().__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        config = BertConfig.from_pretrained("bert-base-uncased")
        with torch.no_grad():
            self.bert = BertModel(config)
            self.dropout = nn.Dropout(p)
            hidden_size = self.bert.pooler.dense.in_features

            self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4) for _ in range(13)])
            self.linear_heads.append(nn.Linear(hidden_size, 2))
            ckpt = torch.load(checkpoint_path, map_location=device)
            from collections import OrderedDict
            new_st = OrderedDict()
            new_st["bert.embeddings.position_ids"] = torch.arange(config.max_position_embeddings).expand((1, -1))
            for key, val in ckpt['model_state_dict'].items():
                if 'bert' in key:
                    nk = key.replace('module.bert.', 'bert.')
                elif 'linear_heads' in key:
                    nk = key.replace('module.linear_heads.', 'linear_heads.')
                new_st[nk] = val
            self.load_state_dict(new_st, strict=False)

        self.to(device)
        self.eval()

    @torch.no_grad()
    def forward(self, reports: List[str]) -> torch.Tensor:
        cleaned = []
        for rep in reports:
            if not isinstance(rep, str):
                rep = "" if rep is None else str(rep)
            cleaned.append(rep.strip().replace("\n", " "))
        tokenized = self.tokenizer(cleaned, padding='longest', truncation=True, max_length=512, return_tensors='pt')
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        out_bert = self.bert(**tokenized)[0]  # [B, seq_len, hidden_dim]
        cls_vec = out_bert[:, 0, :]
        cls_vec = self.dropout(cls_vec)

        preds = []
        for i in range(14):
            out_i = self.linear_heads[i](cls_vec).argmax(dim=1)
            preds.append(out_i)
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def get_cls_vector(self, reports: List[str]) -> torch.Tensor:
        cleaned = []
        for rep in reports:
            if not isinstance(rep, str):
                rep = "" if rep is None else str(rep)
            cleaned.append(rep.strip().replace("\n", " "))
        tokenized = self.tokenizer(cleaned, padding='longest', truncation=True, max_length=512, return_tensors='pt')
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        out_bert = self.bert(**tokenized)[0]
        cls_vec = out_bert[:, 0, :]
        return cls_vec


class CheXbertScorer:
    def __init__(self, checkpoint_path, device='cuda'):
        self.model = CheXbert(checkpoint_path, device)

    def _to_binary(self, preds_14: torch.Tensor) -> torch.Tensor:
        B = preds_14.size(0)
        out = torch.zeros(B, 14, device=preds_14.device)
        for i in range(14):
            out[:, i] = (preds_14[:, i] == 1).float()
        return out

    @torch.no_grad()
    def score_batch_f1(self, ref_texts: list, gen_texts: list, gt_labels: list, batch_size: int = 32) -> torch.Tensor:
        assert len(gen_texts) == len(gt_labels)
        f1_list = []
        for i in range(0, len(gen_texts), batch_size):
            gen_batch = gen_texts[i:i + batch_size]
            ref_batch = ref_texts[i:i + batch_size]
            preds_14 = self.model(gen_batch)
            pred_bin = self._to_binary(preds_14)

            gt_14 = self.model(ref_batch)
            gt_bin = self._to_binary(gt_14)

            tp = (pred_bin * gt_bin).sum(dim=1)
            fp = (pred_bin * (1 - gt_bin)).sum(dim=1)
            fn = ((1 - pred_bin) * gt_bin).sum(dim=1)
            f1_eg = tp / (tp + 0.5 * (fp + fn) + 1e-8)
            f1_list.append(f1_eg)

        f1_tensor = torch.cat(f1_list, dim=0)
        return f1_tensor

    @torch.no_grad()
    def score_batch_f1_with_gt(
            self,
            gen_texts: list,
            gt_labels: list,
            batch_size: int = 32
    ) -> torch.Tensor:
        f1_list = []
        total = len(gen_texts)
        for start_i in range(0, total, batch_size):
            end_i = start_i + batch_size
            batch_gen = gen_texts[start_i:end_i]

            preds_14 = self.model(batch_gen)
            pred_bin = self._to_binary(preds_14)

            gt_14 = torch.tensor(gt_labels[start_i:end_i], dtype=torch.long, device=self.model.device)
            gt_bin = (gt_14 == 1).float()

            tp = (pred_bin * gt_bin).sum(dim=1)
            fp = (pred_bin * (1 - gt_bin)).sum(dim=1)
            fn = ((1 - pred_bin) * gt_bin).sum(dim=1)
            f1_eg = tp / (tp + 0.5 * (fp + fn) + 1e-8)
            f1_list.append(f1_eg)

        return torch.cat(f1_list, dim=0)

    @torch.no_grad()
    def score_batch_similarity(self, ref_texts: List[str], gen_texts: List[str]) -> torch.Tensor:
        ref_vecs = self.model.get_cls_vector(ref_texts)
        gen_vecs = self.model.get_cls_vector(gen_texts)
        return F.cosine_similarity(gen_vecs, ref_vecs, dim=1)


class SwinImageEncoder(nn.Module):
    def __init__(self, model_name, out_dim=768):
        super().__init__()
        self.swin = timm.create_model(model_name, pretrained=True, img_size=512, features_only=True)
        self.proj = nn.Conv2d(768, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor):
        feats_list = self.swin(x)
        feats = feats_list[-1]
        feats = feats.permute(0, 3, 1, 2)
        feats = self.proj(feats)

        B, C, Hf, Wf = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(B, Hf * Wf, C)
        return feats


def build_multi_image_encoder_states(batch_images, encoder, device='cuda', if_frozen=False):
    B = len(batch_images)
    all_imgs = [img for row in batch_images for img in row]
    counts = [len(row) for row in batch_images]
    total_images = len(all_imgs)

    if total_images == 0:
        out_dim = encoder.proj.out_channels
        feats_padded = torch.zeros(B, 1, out_dim, device=device)
        mask_padded = torch.zeros(B, 1, dtype=torch.long, device=device)
        return feats_padded, mask_padded

    big_batch = torch.stack(all_imgs, dim=0).to(device)

    if if_frozen:
        with torch.no_grad():
            feats_ = encoder(big_batch)
    else:
        feats_ = encoder(big_batch)

    splitted = torch.split(feats_, counts, dim=0)
    splitted_flat = [x.reshape(-1, x.size(-1)) for x in splitted]
    feats_padded = pad_sequence(splitted_flat, batch_first=True)

    mask_list = [
        torch.ones(t.size(0), dtype=torch.long, device=device)
        for t in splitted_flat
    ]
    mask_padded = pad_sequence(mask_list, batch_first=True)
    return feats_padded, mask_padded

class MultiImageActor(GPT2LMHeadModel):
    """
    Aggregator for classification only (now binary), plus teacher-forcing with
    ground-truth classification lines in the LM. We also log epoch-level
    (1) total loss, (2) classification loss, and (3) LM loss.
    """

    def __init__(self, vocab_size: int,
                 hidden_dim=768,
                 swin_name="swinv2_tiny_window16_256.ms_in1k",
                 gpt2_name="gpt2"):
        config = GPT2Config.from_pretrained(gpt2_name)
        config.add_cross_attention = True
        config.use_cache = True
        config.vocab_size = vocab_size
        config.n_head = 12
        config.n_embd = 768
        config.n_layer = 30
        super().__init__(config)
        self.encoder = SwinImageEncoder(swin_name, out_dim=hidden_dim)

        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=config.n_head, batch_first=True)
        self.aggregator = nn.TransformerEncoder(layer, num_layers=1)

        # <<< CHANGED: Instead of 14*4 outputs, we now do 14 binary logits
        self.cls_head = nn.Linear(hidden_dim, 14)  # <<< CHANGED

    def forward(self,
                batch_images,
                partial_tokens,
                device='cuda',
                do_sample=False,
                if_frozen=False,
                feats_padded=None,
                mask_padded=None,
                **kwargs):

        if (feats_padded is None) or (mask_padded is None):
            feats_padded, mask_padded = build_multi_image_encoder_states(
                batch_images, self.encoder, device=device, if_frozen=if_frozen
            )

        outputs = super().forward(
            input_ids=partial_tokens.to(device),
            encoder_hidden_states=feats_padded,
            encoder_attention_mask=mask_padded,
            return_dict=True,
            **kwargs
        )
        logits = outputs.logits
        if do_sample:
            last_logits = logits[:, -1, :]
            dist = torch.distributions.Categorical(logits=last_logits)
            nxt = dist.sample()
            lp = dist.log_prob(nxt)
            return nxt, lp
        else:
            return logits




def pad_partials(partials: list, pad_id: int):
    lengths = [p.size(1) for p in partials]
    max_len = max(lengths) if lengths else 1
    B = len(partials)
    device_ = partials[0].device if B > 0 else "cpu"
    out = torch.full((B, max_len), pad_id, dtype=torch.long, device=device_)
    mask = torch.zeros(B, max_len, dtype=torch.long, device=device_)
    for i, seq in enumerate(partials):
        seq_len = seq.size(1)
        out[i, :seq_len] = seq[0]
        mask[i, :seq_len] = 1
    return out, mask


class GPT2BeamSearchWrapper(GPT2LMHeadModel):
    def __init__(self, actor):
        super().__init__(actor.config)
        self.transformer = actor.transformer
        self.lm_head = actor.lm_head
        self.config = actor.config

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            **kwargs
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


def get_reference_text(sample: dict) -> str:
    def safe_str(x):
        if isinstance(x, str):
            return x.strip()
        if not x or isinstance(x, float):
            return ""
        return str(x).strip()

    findings_val = safe_str(sample.get("findings", ""))
    impression_val = safe_str(sample.get("impression", ""))

    if findings_val and impression_val:
        return f"FINDINGS: {findings_val} IMPRESSION: {impression_val}"
    else:
        return f"FINDINGS: {findings_val}"


special_token_2_id = {
    "<Blank>": 0,
    "<Positive>": 1,
    "<Negative>": 2,
    "<Uncertain>": 3
}


def map_token_to_class_id(token_str: str) -> int:
    return special_token_2_id.get(token_str, 0)


# -----------------------------------------------------------------------------
#                         ** CHANGED SECTION START **
# -----------------------------------------------------------------------------
def compute_pos_only_f1(preds_14: torch.Tensor, gold_14: torch.Tensor, eps=1e-8) -> float:
    """
    For binary classification, we interpret 'preds_14' as 0/1 predictions
    and 'gold_14' as 0/1 gold. So we compute "positive-only" F1 across all conditions.

    NOTE: The user might want a multi-label approach. This function
    uses thresholded predictions => we assume preds_14 is already {0,1}.
    """
    pred_bin = (preds_14 == 1).float()
    gold_bin = (gold_14 == 1).float()

    tp = (pred_bin * gold_bin).sum(dim=1)
    fp = (pred_bin * (1 - gold_bin)).sum(dim=1)
    fn = ((1 - pred_bin) * gold_bin).sum(dim=1)

    f1_eg = tp / (tp + 0.5 * (fp + fn) + eps)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(f1_eg.mean().item()), float(precision.mean().item()), float(recall.mean().item())


# -----------------------------------------------------------------------------
#                         ** CHANGED SECTION END **
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(
        actor,
        data_loader,
        tokenizer,
        device,
        chex_scorer,
        use_report_tags: bool = True,
        max_seq_len=32,
        test_subset_frac=0.2,
        num_beams=1
):
    ddp = isinstance(actor, DDP)
    base_actor = actor.module if ddp else actor

    hf_model = GPT2BeamSearchWrapper(base_actor)
    hf_model.eval()
    hf_model.to(device)

    all_refs = []
    all_gens = []
    all_impress_gens = []
    all_chex_gt = []

    all_class_preds = []
    all_class_gts = []

    total_test = len(data_loader)
    max_test_batches = min(
        total_test,
        max(1, math.ceil(test_subset_frac * total_test))
    )
    ddp_rank0 = (not ddp) or (ddp and int(os.environ.get('RANK', '0')) == 0)
    if ddp_rank0:
        iterator = tqdm(enumerate(data_loader),
                        total=min(max_test_batches, total_test),
                        desc="Evaluating (Beam)")
    else:
        iterator = enumerate(data_loader)

    eos_id = tokenizer.eos_token_id or tokenizer.bos_token_id

    for b_i, batch_data in iterator:
        if b_i >= max_test_batches:
            break
        B = len(batch_data["report_all"])
        if B == 0:
            continue

        ref_texts = []
        for i in range(B):
            findings_ = batch_data["findings"][i]
            impress_ = batch_data["impression"][i]
            if findings_ and impress_:
                ref_str = f"FINDINGS: {findings_} IMPRESSION: {impress_}"
            else:
                ref_str = f"FINDINGS: {findings_}"
            ref_texts.append(ref_str)
        all_refs.extend(ref_texts)

        all_chex_gt.extend(batch_data["chexbert_labels"])

        feats_padded, mask_padded = build_multi_image_encoder_states(
            batch_data["images"], base_actor.encoder, device=device, if_frozen=True,
        )
        aggregator_out = base_actor.aggregator(
            feats_padded,
            src_key_padding_mask=(mask_padded == 0)
        )
        mask_float = mask_padded.float().unsqueeze(-1)
        feats_sum = (aggregator_out * mask_float).sum(dim=1)
        denom = mask_float.sum(dim=1).clamp_min(1e-8)
        feats_pooled = feats_sum / denom

        # <<< CHANGED: aggregator => 14 binary logits
        cls_logits = base_actor.cls_head(feats_pooled)  # shape [B,14]
        # Threshold at 0.5 => predicted label
        preds_bin = (torch.sigmoid(cls_logits) >= 0.5).long()  # <<< CHANGED
        all_class_preds.append(preds_bin.cpu())

        # Also remap the ground-truth 0..3 => binary
        gt_labels_full = torch.tensor(batch_data["chexbert_labels"], dtype=torch.long)
        gt_bin = (gt_labels_full == 1).long()  # 1 => positive, else => 0
        all_class_gts.append(gt_bin)

        partial_inputs = []
        for i in range(B):
            row = preds_bin[i]
            parts = []
            for idx_cond, cond_name in enumerate(CONDITIONS):
                # if row[idx_cond]==1 => Positive else Negative
                class_str = "Positive" if row[idx_cond].item() == 1 else "Blank"
                cond_token = cond_name.replace(" ", "_")
                parts.append(f"<{cond_token}>:<{class_str}>")
            classification_line = ";".join(parts) + " <cls_end>"
            partial_inputs.append(classification_line)

        prefix_text = "<findings> Findings:" if use_report_tags else "Findings:"
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        input_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)
        enc = tokenizer(partial_inputs, padding='longest', return_tensors='pt').to(device)
        generated = hf_model.generate(
            input_ids=input_ids, #enc["input_ids"],
            # attention_mask=enc["attention_mask"],
            encoder_hidden_states=feats_padded,
            encoder_attention_mask=mask_padded,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_seq_len,
            do_sample=False,
            use_cache=False,
            num_beams=num_beams
        )

        gen_texts = []
        for i in range(generated.size(0)):
            # prompt_len_i = enc["attention_mask"][i].sum().item()
            new_tokens = generated[i, :] #prompt_len_i:]
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            gen_texts.append(new_text)
        all_gens.extend(gen_texts)

        for gtxt in gen_texts:
            match = re.search(r"Impression:\s*(.*)", gtxt, re.IGNORECASE)
            if match:
                only_impr = match.group(1).strip()
            else:
                only_impr = ""
            all_impress_gens.append(only_impr)

    if len(all_class_preds) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    all_class_preds_tensor = torch.cat(all_class_preds, dim=0)
    all_class_gts_tensor = torch.cat(all_class_gts, dim=0)

    classification_f1, p_, r_ = compute_pos_only_f1(
        all_class_preds_tensor, all_class_gts_tensor
    )

    nlg_dict = compute_batch_nlg_metrics(all_refs, all_gens, num_proc=4)

    f1_vals_full = chex_scorer.score_batch_f1(all_refs, all_gens, all_chex_gt)
    text_based_f1 = float(f1_vals_full.mean().item())

    f1_vals_impr = chex_scorer.score_batch_f1_with_gt(all_impress_gens, all_chex_gt)
    impression_f1 = float(f1_vals_impr.mean().item())

    if ddp_rank0:
        logging.info(f"[Eval => aggregator Classification F1={classification_f1:.3f}, "
                     f"Precision={p_:.3f}, Recall={r_:.3f}]")
        logging.info(f"[Eval => Full text] BLEU4={nlg_dict['BLEU_4']:.3f}, "
                     f"METEOR={nlg_dict['METEOR']:.3f}, ROUGE_L={nlg_dict['ROUGE_L']:.3f}, "
                     f"Text-F1(full)={text_based_f1:.3f}, Text-F1(impression)={impression_f1:.3f}")

        for i in range(min(2, len(all_gens))):
            logging.info(f"--- Example {i} ---")
            logging.info(f"Prompt: {partial_inputs[i]}")
            logging.info(f"Generated (full): {all_gens[i]}")
            logging.info(f"Generated (only impression): {all_impress_gens[i]}")
            logging.info(f"Ref: {all_refs[i]}")

    return classification_f1, nlg_dict["BLEU_1"], nlg_dict["BLEU_4"], nlg_dict["METEOR"], nlg_dict["ROUGE_L"], text_based_f1


@torch.no_grad()
def evaluate_classifier_only(
        actor,
        data_loader,
        device,
        test_subset_frac=0.2
):
    ddp = isinstance(actor, DDP)
    base_actor = actor.module if ddp else actor
    base_actor.eval()
    base_actor.to(device)

    all_class_preds = []
    all_class_gts = []

    total_test = len(data_loader)
    max_test_batches = min(
        total_test,
        max(1, math.ceil(test_subset_frac * total_test))
    )
    ddp_rank0 = (not ddp) or (ddp and int(os.environ.get('RANK', '0')) == 0)

    if ddp_rank0:
        iterator = tqdm(enumerate(data_loader),
                        total=min(max_test_batches, total_test),
                        desc="Evaluating Classifier Only")
    else:
        iterator = enumerate(data_loader)

    for b_i, batch_data in iterator:
        if b_i >= max_test_batches:
            break

        B = len(batch_data["report_all"])
        if B == 0:
            continue

        feats_padded, mask_padded = build_multi_image_encoder_states(
            batch_data["images"], base_actor.encoder, device=device, if_frozen=True,
        )
        aggregator_out = base_actor.aggregator(
            feats_padded,
            src_key_padding_mask=(mask_padded == 0)
        )
        mask_float = mask_padded.float().unsqueeze(-1)
        feats_sum = (aggregator_out * mask_float).sum(dim=1)
        denom = mask_float.sum(dim=1).clamp_min(1e-8)
        feats_pooled = feats_sum / denom

        # <<< CHANGED: aggregator => 14 binary logits
        cls_logits = base_actor.cls_head(feats_pooled)  # shape [B,14]
        # threshold at 0.5
        preds_bin = (torch.sigmoid(cls_logits) >= 0.5).long()  # <<< CHANGED
        all_class_preds.append(preds_bin.cpu())

        # Also remap the ground-truth 0..3 => binary
        gt_labels = torch.tensor(batch_data["chexbert_labels"], dtype=torch.long)
        gt_bin = (gt_labels == 1).long()  # <<< CHANGED
        all_class_gts.append(gt_bin)

    if len(all_class_preds) == 0:
        return 0.0, 0.0, 0.0

    all_class_preds_tensor = torch.cat(all_class_preds, dim=0)
    all_class_gts_tensor = torch.cat(all_class_gts, dim=0)

    classification_f1, precision, recall = compute_pos_only_f1(
        all_class_preds_tensor, all_class_gts_tensor
    )

    if ddp_rank0:
        logging.info(
            f"[Classifier-Only Eval] Aggregator Classification: "
            f"F1={classification_f1:.3f}, P={precision:.3f}, R={recall:.3f}"
        )

    return classification_f1, precision, recall


def resume_checkpoint(
        path: str,
        model: nn.Module,
        optimizer=None,
        ddp=False,
        device='cuda'
) -> int:
    if (not path) or (not os.path.isfile(path)):
        return 0
    ckpt = torch.load(path, map_location=device)
    logging.info(f"[Resume] Loading from {path} ...")
    if "actor_state" in ckpt:
        state_dict = ckpt["actor_state"]
    elif "critic_state" in ckpt:
        state_dict = ckpt["critic_state"]
    else:
        logging.info("Checkpoint has neither 'actor_state' nor 'critic_state'. Skipping load.")
        return 0
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            new_k = k[len(unwanted_prefix):]
            state_dict[new_k] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=True)
    if optimizer:
        if "actor_optim" in ckpt:
            optimizer.load_state_dict(ckpt["actor_optim"])
        elif "critic_optim" in ckpt:
            optimizer.load_state_dict(ckpt["critic_optim"])
    last_cycle = ckpt.get("last_cycle", ckpt.get("last_cycle", 0))
    pos_weight = None
    if "pos_weight" in ckpt:
        pos_weight = ckpt["pos_weight"].to(device)
        print("[INFO] pos_weight loaded from checkpoint")
    logging.info(f"... resume done. last_epoch={last_cycle}")
    return last_cycle, pos_weight


def compute_pos_weight_from_csv(csv_path, split_mode="train", filter_findings=True, max_images=3):
    """
    Reads the CSV once, filters rows for training, then
    parses 'chexbert_labels' to count positives/negatives for each condition.
    Returns a per-condition pos_weight as a PyTorch Tensor of shape [14].
    """
    # -- Read CSV --
    df = pd.read_csv(csv_path)

    # Example: filter by split == "train" or "train"+"validate"
    if split_mode == "train":
        df = df[df["split"].isin(["train", "validate"])]
    elif split_mode == "test":
        df = df[df["split"] == "test"]
    else:
        raise ValueError("split_mode must be 'train' or 'test'")

    # (Optional) filter out rows with > max_images
    valid_rows = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        image_paths = row["image_paths"].split(";")
        if len(image_paths) <= max_images:
            valid_rows.append(row)
    df = pd.DataFrame(valid_rows) if len(valid_rows) > 0 else pd.DataFrame(columns=df.columns)

    print(f"[INFO] After filtering, {len(df)} rows for split_mode='{split_mode}'.")

    # -- Initialize counters for each of 14 conditions --
    num_conditions = 14
    pos_count = np.zeros(num_conditions, dtype=np.float32)
    neg_count = np.zeros(num_conditions, dtype=np.float32)

    # -- Parse each row's chexbert_labels --
    for label_str in df["chexbert_labels"]:
        # e.g. label_str = "1;0;2;3;..." (14 values)
        label_ints = [int(x) for x in label_str.split(";")]  # length=14
        # Convert each int => 1 if x==1 else 0
        bin_labels = [1 if x == 1 else 0 for x in label_ints]
        for i, val in enumerate(bin_labels):
            if val == 1:
                pos_count[i] += 1
            else:
                neg_count[i] += 1

    # -- Compute pos_weight[i] = neg_count[i] / (pos_count[i] + eps) --
    epsilon = 1e-8
    pos_weight_np = neg_count / (pos_count + epsilon)

    print("[INFO] pos_count =", pos_count)
    print("[INFO] neg_count =", neg_count)
    print("[INFO] pos_weight =", pos_weight_np)

    # Return as a PyTorch Tensor
    return torch.from_numpy(pos_weight_np)


def main(args):
    ddp, device, master_process, ddp_world_size = setup_distributed(args)
    import numpy as np

    if args.warmup_actor_effective_batch_size is None:
        args.warmup_actor_effective_batch_size = args.warmup_actor_batch_size
    if args.warmup_actor_effective_batch_size <= 0:
        raise ValueError("warmup_actor_effective_batch_size must be positive.")

    ckpt_dir = args.ckpt_dir if args.ckpt_dir else "."
    args.ckpt_dir = ckpt_dir

    if master_process:
        logger, logfile = setup_logger(args.log_dir)
        logger.info("Starting the training script...")
        logger.info(f"Log file: {os.path.abspath(logfile)}")
        os.makedirs(ckpt_dir, exist_ok=True)
        logger.info(f"Checkpoint directory: {os.path.abspath(ckpt_dir)}")
    else:
        if args.ckpt_dir and args.ckpt_dir != ".":
            os.makedirs(args.ckpt_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    chex_condition_specials = []
    for cond in CONDITIONS:
        cond_token = cond.replace(" ", "_")
        special_tok = f"<{cond_token}>"
        if special_tok not in tokenizer.get_vocab():
            chex_condition_specials.append(special_tok)

    chex_class_specials = []
    for class_str in CLASS_MAPPING.values():
        clz_token = class_str.replace(" ", "_")
        special_tok = f"<{clz_token}>"
        if special_tok not in tokenizer.get_vocab():
            chex_class_specials.append(special_tok)

    additional_specials = []
    if "<cls_end>" not in tokenizer.get_vocab():
        additional_specials.append("<cls_end>")
    if args.use_report_tags:
        for tag_tok in ("<next>", "<impression>", "<findings>"):
            if tag_tok not in tokenizer.get_vocab():
                additional_specials.append(tag_tok)

    all_new_specials = chex_condition_specials + chex_class_specials + additional_specials
    if len(all_new_specials) > 0:
        tokenizer.add_special_tokens({"additional_special_tokens": all_new_specials})

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    transform_train = T.Compose([
        T.Resize(args.image_size + 64),
        T.RandomCrop(args.image_size),
        T.RandomRotation(degrees=5),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = T.Compose([
        T.Resize(args.image_size + 64),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    ptdtype = {'float32': torch.float32,
               'float16': torch.float16,
               'bfloat16': torch.bfloat16}[args.dtype]

    train_language_dataset = MIMICCXRDataset(args.csv_path, transform=transform_train,
                                    split_mode="train", filter_findings=args.filter_findings,
                                    max_images=args.max_images)
    train_image_dataset = MIMICCXRIMAGEDataset(args.csv_path, transform=transform_train,
                                    split_mode="train", filter_findings=args.filter_findings,
                                    max_images=3)
    test_dataset = MIMICCXRDataset(args.csv_path, transform=transform_test,
                                   split_mode="test", filter_findings=args.filter_findings,
                                   max_images=20)

    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler_train = DistributedSampler(
            train_language_dataset,
            shuffle=True,
            num_replicas=ddp_world_size,
            rank=int(os.environ['RANK']),
            drop_last=True
        )
        sampler_train_image = DistributedSampler(
            train_image_dataset,
            shuffle=True,
            num_replicas=ddp_world_size,
            rank=int(os.environ['RANK']),
            drop_last=True
        )
    else:
        sampler_train = None
        sampler_train_image = None

    train_warmup_language_loader = DataLoader(
        train_language_dataset,
        batch_size=args.warmup_actor_batch_size,
        shuffle=(sampler_train is None),
        sampler=sampler_train,
        collate_fn=my_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader_with_train_dataset = DataLoader(
        train_language_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=my_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    train_warmup_image_loader = DataLoader(
        train_image_dataset,
        batch_size=args.warmup_actor_batch_size,
        shuffle=(sampler_train_image is None),
        sampler=sampler_train_image,
        collate_fn=my_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=my_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    pos_count = torch.zeros(14)
    neg_count = torch.zeros(14)

    actor = MultiImageActor(vocab_size=len(tokenizer))
    actor.resize_token_embeddings(len(tokenizer))
    logging.info(
        f"[Param Counts] actor total={sum(p.numel() for p in actor.parameters() if p.requires_grad) / 1e6:.2f}M")
    logging.info(
        f"[Param Counts] actor swin={sum(p.numel() for p in actor.encoder.parameters() if p.requires_grad) / 1e6:.2f}M")
    logging.info(
        f"[Param Counts] actor aggregator={sum(p.numel() for p in actor.aggregator.parameters() if p.requires_grad) / 1e6:.2f}M")
    logging.info(
        f"[Param Counts] actor gpt={sum(p.numel() for p in actor.transformer.parameters() if p.requires_grad) / 1e6:.2f}M")

    actor.to(device)

    # <<< CHANGED: Use BCEWithLogitsLoss for binary multi-label classification

    actor_optimA = optim.RAdam(actor.parameters(), lr=args.warmup_actor_lr)

    last_cycle = 0
    if args.resume_actor_path:
        last_cycle, pos_weight = resume_checkpoint(
            args.resume_actor_path,
            actor,
            optimizer=actor_optimA,
            ddp=ddp,
            device=device
        )

    pos_weight = compute_pos_weight_from_csv(
        args.csv_path,
        split_mode="train",
        filter_findings=True,
        max_images=args.max_images,
    )

    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    if args.compile_model and master_process:
        logger.info("Compiling models...")
    if args.compile_model:
        actor = torch.compile(actor)

    ddp_flag = ddp
    if ddp_flag:
        actor = DDP(actor, device_ids=[int(os.environ['LOCAL_RANK'])], find_unused_parameters=True)

    raw_actor = actor.module if ddp_flag else actor

    chex_scorer = CheXbertScorer(args.chexbert_path, device=device)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))

    # if master_process:
    #     actor.eval()
    #     classification_f1, b1, b4, mr, rg, f1v = evaluate_model(
    #         actor,
    #         test_loader,
    #         tokenizer,
    #         device,
    #         chex_scorer,
    #         max_seq_len=args.max_seq_len_test,
    #         test_subset_frac=args.test_subset_frac,
    #         num_beams=1
    #     )
    #     logging.info(
    #         f"[Test Set Eval aggregator-for-class-only => Classification F1={classification_f1:.3f}, "
    #         f"BLEU1={b1:.3f}, BLEU4={b4:.3f}, METEOR={mr:.3f}, ROUGE_L={rg:.3f}, F1={f1v:.3f}"
    #     )
    #
    #     classification_f1, b1, b4, mr, rg, f1v = evaluate_model(
    #         actor,
    #         test_loader_with_train_dataset,
    #         tokenizer,
    #         device,
    #         chex_scorer,
    #         max_seq_len=args.max_seq_len_test,
    #         test_subset_frac=0.01,
    #         num_beams=1
    #     )
    #     logging.info(
    #         f"[Train Set Eval aggregator-for-class-only => Classification F1={classification_f1:.3f}, "
    #         f"BLEU1={b1:.3f}, BLEU4={b4:.3f}, METEOR={mr:.3f}, ROUGE_L={rg:.3f}, F1={f1v:.3f}"
    #     )
    # torch.distributed.barrier()
    if args.eval_only:
        if master_process:
            # classification_f1, b4, mr, rg, f1v = evaluate_model(
            #     actor,
            #     test_loader,
            #     tokenizer,
            #     device,
            #     chex_scorer,
            #     max_seq_len=args.max_seq_len_test,
            #     test_subset_frac=args.test_subset_frac,
            #     num_beams=1
            # )
            # logging.info(
            #     f"[Test Set Eval aggregator-for-class-only => Classification F1={classification_f1:.3f}, "
            #     f"BLEU4={b4:.3f}, METEOR={mr:.3f}, ROUGE_L={rg:.3f}, F1={f1v:.3f}"
            # )

            classification_f1, b4, mr, rg, f1v = evaluate_model(
                actor,
                test_loader_with_train_dataset,
                tokenizer,
                device,
                chex_scorer,
                max_seq_len=args.max_seq_len_test,
                test_subset_frac=0.005,
                num_beams=1
            )
            logging.info(
                f"[Train Set Eval aggregator-for-class-only => Classification F1={classification_f1:.3f}, "
                f"BLEU4={b4:.3f}, METEOR={mr:.3f}, ROUGE_L={rg:.3f}, F1={f1v:.3f}"
            )
    else:
        outer_iter = 0
        for cyc_i in range(last_cycle, args.pretrain_cycles):
            cur_lr = get_lr(
                cyc_i,
                args.warmup_iters_pretrain,
                args.lr_decay_iters_pretrain,
                args.learning_rate,
                args.min_lr
            )

            # update both actor_optimA and critic_optimB
            for pg in actor_optimA.param_groups:
                pg["lr"] = cur_lr

            if master_process:
                logging.info(f"[Pretrain Cycle {cyc_i + 1}/{args.pretrain_cycles}] => LR={cur_lr:.6g}")

            if args.actor_warmup_epochs > 0:
                global_step = 0
                print_every = 400
                for ep in range(args.actor_warmup_epochs):
                    actor.train()
                    if ddp_flag and hasattr(train_warmup_language_loader.sampler, 'set_epoch'):
                        train_warmup_language_loader.sampler.set_epoch(ep)
                        train_warmup_image_loader.sampler.set_epoch(ep)

                    e_desc = f"Actor Warmup (Cycle {cyc_i + 1}) Epoch {ep + 1}/{args.actor_warmup_epochs}"
                    if master_process:
                        if cyc_i < args.image_train_cyc:
                            total_batches = len(train_warmup_image_loader)
                            max_batches = min(
                                total_batches,
                                max(1, math.ceil(args.train_actor_subset_frac * total_batches))
                            )
                            iterator = tqdm(enumerate(train_warmup_image_loader),
                                            total=min(max_batches, total_batches),
                                            desc=e_desc)
                        else:
                            total_batches = len(train_warmup_language_loader)
                            max_batches = min(
                                total_batches,
                                max(1, math.ceil(args.train_actor_subset_frac * total_batches))
                            )
                            iterator = tqdm(enumerate(train_warmup_language_loader),
                                            total=min(max_batches, total_batches),
                                            desc=e_desc)

                    else:
                        if cyc_i < args.image_train_cyc:
                            total_batches = len(train_warmup_image_loader)
                            max_batches = min(
                                total_batches,
                                max(1, math.ceil(args.train_actor_subset_frac * total_batches))
                            )
                            iterator = enumerate(train_warmup_image_loader)
                        else:
                            total_batches = len(train_warmup_language_loader)
                            max_batches = min(
                                total_batches,
                                max(1, math.ceil(args.train_actor_subset_frac * total_batches))
                            )
                            iterator = enumerate(train_warmup_language_loader)

                    epoch_loss_sum_total = 0.0
                    epoch_loss_sum_cls = 0.0
                    epoch_loss_sum_lm = 0.0
                    epoch_f1_sum = 0.0  # (F1) <<< ADDED >>>
                    epoch_count = 0

                    for b_i, batch_data in iterator:
                        if b_i >= max_batches:
                            break

                        B = len(batch_data["report_all"])
                        if B == 0:
                            continue

                        micro_bs = math.ceil(B / args.warmup_actor_grad_accum)
                        start_i = 0

                        step_loss_sum_total = 0.0
                        step_loss_sum_cls = 0.0
                        step_loss_sum_lm = 0.0
                        step_f1_sum = 0.0
                        micro_batches_processed = 0

                        actor_optimA.zero_grad(set_to_none=True)

                        effective_scale = float(ddp_world_size) / float(args.warmup_actor_effective_batch_size)

                        for micro_step in range(args.warmup_actor_grad_accum):
                            end_i = min(start_i + micro_bs, B)
                            if start_i >= end_i:
                                break

                            mb_imgs = batch_data["images"][start_i:end_i]
                            mb_findings = batch_data["findings"][start_i:end_i]
                            mb_impress = batch_data["impression"][start_i:end_i]
                            mb_labels = torch.tensor(
                                batch_data["chexbert_labels"][start_i:end_i],
                                dtype=torch.long, device=device
                            )
                            mb_size = len(mb_imgs)
                            start_i = end_i

                            if mb_size == 0:
                                continue

                            is_last_micro = (end_i >= B)
                            if ddp_flag:
                                actor.require_backward_grad_sync = is_last_micro

                            micro_batches_processed += 1

                            with (torch.amp.autocast('cuda', dtype=ptdtype)
                                  if device != 'cpu' else nullcontext()):

                                # Optional: combine with LM or do classification-only
                                if cyc_i < args.image_train_cyc:
                                    feats_padded, mask_padded = build_multi_image_encoder_states(
                                        mb_imgs, raw_actor.encoder, device=device, if_frozen=False,
                                    )
                                    aggregator_out = raw_actor.aggregator(
                                        feats_padded,
                                        src_key_padding_mask=(mask_padded == 0)
                                    )
                                    mask_float = mask_padded.float().unsqueeze(-1)
                                    feats_sum = (aggregator_out * mask_float).sum(dim=1)
                                    denom = mask_float.sum(dim=1).clamp_min(1e-8)
                                    feats_pooled = feats_sum / denom

                                    # <<< CHANGED: aggregator => 14
                                    logits_14 = raw_actor.cls_head(feats_pooled)  # shape [mb_size,14]
                                    # remap ground-truth => binary
                                    mb_labels_bin = (mb_labels == 1).float()  # <<< CHANGED
                                    # BCE
                                    loss_cls_ = criterion_cls(logits_14, mb_labels_bin)  # <<< CHANGED
                                    # Just classification
                                    loss_lm_ = torch.tensor(0.0, device=device)
                                    loss_total = loss_cls_
                                    loss_scale = float(mb_size) * effective_scale
                                    loss_val = loss_total * loss_scale
                                    with torch.no_grad():
                                        preds_bin = (torch.sigmoid(
                                            logits_14) >= 0.5).long()  # shape [mb_size,14], in {0,1}
                                        batch_f1_val, _, _ = compute_pos_only_f1(preds_bin, mb_labels_bin.long())
                                        step_f1_sum += batch_f1_val  # average F1 for this micro-batch
                                else:
                                    # incorporate LM as well
                                    feats_padded, mask_padded = build_multi_image_encoder_states(
                                        mb_imgs, raw_actor.encoder, device=device, if_frozen=False,
                                    )
                                    aggregator_out = raw_actor.aggregator(
                                        feats_padded,
                                        src_key_padding_mask=(mask_padded == 0)
                                    )
                                    mask_float = mask_padded.float().unsqueeze(-1)
                                    feats_sum = (aggregator_out * mask_float).sum(dim=1)
                                    denom = mask_float.sum(dim=1).clamp_min(1e-8)
                                    feats_pooled = feats_sum / denom

                                    # <<< CHANGED: aggregator => 14
                                    logits_14 = raw_actor.cls_head(feats_pooled)  # shape [mb_size,14]
                                    # remap ground-truth => binary
                                    mb_labels_bin = (mb_labels == 1).float()  # <<< CHANGED
                                    # BCE
                                    loss_cls_ = criterion_cls(logits_14, mb_labels_bin)  # <<< CHANGED
                                    with torch.no_grad():
                                        preds_bin = (torch.sigmoid(
                                            logits_14) >= 0.5).long()  # shape [mb_size,14], in {0,1}
                                        batch_f1_val, _, _ = compute_pos_only_f1(preds_bin, mb_labels_bin.long())
                                        step_f1_sum += batch_f1_val  # average F1 for this micro-batch
                                    final_inputs = []
                                    for i_s in range(mb_size):
                                        sample_dict = {
                                            "chexbert_labels": batch_data["chexbert_labels"][start_i - mb_size + i_s],
                                            "findings": mb_findings[i_s],
                                            "impression": mb_impress[i_s],
                                        }
                                        ref_str = build_training_reference(
                                            sample_dict,
                                            shuffle_findings=args.shuffle_findings,
                                            use_report_tags=args.use_report_tags
                                        )
                                        ref_str += tokenizer.eos_token
                                        final_inputs.append(ref_str)

                                    enc_inputs = tokenizer(
                                        final_inputs,
                                        padding='longest',
                                        truncation=True,
                                        return_tensors='pt'
                                    ).to(device)

                                    input_ids_ = enc_inputs["input_ids"]
                                    attn_ = enc_inputs["attention_mask"]

                                    cls_end_tok = tokenizer.convert_tokens_to_ids("<cls_end>")
                                    exclude_mask = torch.ones_like(input_ids_, device=device)
                                    for i_s in range(mb_size):
                                        row_ids = input_ids_[i_s]
                                        pos_end = (row_ids == cls_end_tok).nonzero()
                                        if pos_end.size(0) > 0:
                                            first_occ = pos_end[0].item()
                                            exclude_mask[i_s, :first_occ + 1] = 0

                                    lm_out = raw_actor(
                                        None,
                                        input_ids_,
                                        device=device,
                                        feats_padded=feats_padded,
                                        mask_padded=mask_padded
                                    )
                                    shift_logits = lm_out[:, :-1, :].contiguous()
                                    shift_labels = input_ids_[:, 1:].contiguous()
                                    shift_attn = attn_[:, 1:].contiguous()
                                    shift_exc = exclude_mask[:, 1:].contiguous()
                                    final_mask = (shift_attn * shift_exc).bool()
                                    shift_labels_m = shift_labels.masked_fill(~final_mask, -100)

                                    loss_lm_ = F.cross_entropy(
                                        shift_logits.view(-1, shift_logits.size(-1)),
                                        shift_labels_m.view(-1),
                                        ignore_index=-100
                                    )
                                    loss_total = loss_lm_ + loss_cls_
                                    loss_scale = float(mb_size) * effective_scale
                                    loss_val = loss_total * loss_scale
                            if scaler:
                                scaler.scale(loss_val).backward()
                            else:
                                loss_val.backward()

                            step_loss_sum_cls += float(loss_cls_.item())
                            step_loss_sum_lm += float(loss_lm_.item())
                            step_loss_sum_total += float(loss_val.item())

                        if scaler:
                            scaler.unscale_(actor_optimA)
                            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                            scaler.step(actor_optimA)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                            actor_optimA.step()

                        actor_optimA.zero_grad(set_to_none=True)
                        global_step += 1

                        if (global_step % print_every == 0) and master_process:
                            avg_step_f1 = step_f1_sum / max(1, micro_batches_processed)
                            logging.info(
                                f"[Actor Warmup] global_step={global_step}, "
                                f"partial_total_loss_sum={step_loss_sum_total:.4f}, "
                                f"cls_loss={step_loss_sum_cls:.4f}, lm_loss={step_loss_sum_lm:.4f}, "
                                f"train_f1={avg_step_f1:.4f}"
                            )

                        epoch_loss_sum_total += step_loss_sum_total
                        epoch_loss_sum_cls += step_loss_sum_cls
                        epoch_loss_sum_lm += step_loss_sum_lm
                        epoch_f1_sum += (step_f1_sum / max(1, micro_batches_processed))
                        epoch_count += 1

                    if master_process and epoch_count > 0:
                        avg_ep_loss_total = epoch_loss_sum_total / epoch_count
                        avg_ep_loss_cls = epoch_loss_sum_cls / epoch_count
                        avg_ep_loss_lm = epoch_loss_sum_lm / epoch_count
                        avg_ep_f1 = epoch_f1_sum / epoch_count  # (F1) <<< ADDED >>>
                        logging.info(
                            f"[Actor Warmup: cycle={cyc_i + 1}, epoch={ep + 1}] => "
                            f"avg_total={avg_ep_loss_total:.4f}, avg_cls={avg_ep_loss_cls:.4f}, "
                            f"avg_lm={avg_ep_loss_lm:.4f}, train_f1={avg_ep_f1:.4f}"
                        )

            if master_process and ((cyc_i + 1) % args.save_every_warmup == 0):
                ckpt_prefix = f"actor_warmup_cycle_{cyc_i + 1}_larger"
                ckpt_filename = f"{ckpt_prefix}_epoch_{args.actor_warmup_epochs}.pth"
                ckpt_path = os.path.join(args.ckpt_dir, ckpt_filename)
                tosave = {
                    "actor_state": raw_actor.state_dict(),
                    "actor_optim": actor_optimA.state_dict(),
                    "last_epoch": args.actor_warmup_epochs,
                    'last_cycle': cyc_i + 1,
                    "pos_weight": pos_weight.cpu(),  # store on CPU
                }
                torch.save(tosave, ckpt_path)
                logging.info(f"[Actor Warmup] checkpoint saved => {os.path.abspath(ckpt_path)}")

            if master_process and (cyc_i < args.image_train_cyc):
                f1, precision, recall = evaluate_classifier_only(
                    actor,
                    test_loader,
                    device,
                    test_subset_frac=args.test_subset_frac)
                logging.info(
                    f"[Eval aggregator-for-class-only => Classification F1={f1:.3f}, "
                    f"Precision={precision:.3f}, Recall={recall:.3f}]"
                )

            if master_process and ((cyc_i + 1) % args.eval_every_pretrain == 0) and (cyc_i > args.image_train_cyc):
                actor.eval()
                classification_f1, b1, b4, mr, rg, f1v = evaluate_model(
                    actor,
                    test_loader,
                    tokenizer,
                    device,
                    chex_scorer,
                    use_report_tags=args.use_report_tags,
                    max_seq_len=args.max_seq_len_test,
                    test_subset_frac=args.test_subset_frac,
                    num_beams=1
                )
                logging.info(
                    f"[Test Set Eval aggregator-for-class-only => Classification F1={classification_f1:.3f}, "
                    f"BLEU1={b1:.3f}, BLEU4={b4:.3f}, METEOR={mr:.3f}, ROUGE_L={rg:.3f}, F1={f1v:.3f}]"
                )

                classification_f1, b1, b4, mr, rg, f1v = evaluate_model(
                    actor,
                    test_loader_with_train_dataset,
                    tokenizer,
                    device,
                    chex_scorer,
                    use_report_tags=args.use_report_tags,
                    max_seq_len=args.max_seq_len_test,
                    test_subset_frac=0.01,
                    num_beams=1
                )
                logging.info(
                    f"[Train Set Eval aggregator-for-class-only => Classification F1={classification_f1:.3f}, "
                    f"BLEU1={b1:.3f}, BLEU4={b4:.3f}, METEOR={mr:.3f}, ROUGE_L={rg:.3f}, F1={f1v:.3f}"
                )

            outer_iter += 1

    if ddp_flag:
        cleanup_distributed()

    if master_process:
        logging.info("All training done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, default="PREPROCESSED DATA FILE PATH")
    parser.add_argument("--chexbert_path", type=str, default="CHEXBERT CHECKPINT PATH")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--warmup_actor_batch_size", type=int, default=128)
    parser.add_argument("--warmup_actor_grad_accum", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--log_dir", type=str, default="./log",
                        help="Directory to write training logs. Defaults to current working directory.")
    parser.add_argument("--ckpt_dir", type=str, default="./log",
                        help="Directory to store warmup checkpoints. Defaults to current working directory.")
    parser.add_argument(
        "--warmup_actor_effective_batch_size",
        type=int,
        default=None,
        help="Target effective global batch size for actor warmup steps. Defaults to warmup_actor_batch_size when unset.",
    )
    parser.add_argument("--allow_tf32", action="store_true", default=True)
    parser.add_argument("--compile_model", action="store_true", default=True)

    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])

    parser.add_argument("--eval_only", action="store_true", default=False)

    parser.add_argument("--pretrain_cycles", type=int, default=70)
    parser.add_argument("--actor_warmup_epochs", type=int, default=1)
    parser.add_argument("--warmup_actor_lr", type=float, default=1e-4)
    parser.add_argument("--resume_actor_path", type=str, default='')
    parser.add_argument("--save_every_warmup", type=int, default=1)
    parser.add_argument("--eval_every_pretrain", type=int, default=5)

    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_seq_len_test", type=int, default=512)
    parser.add_argument("--train_actor_subset_frac", type=float, default=1.0)
    parser.add_argument("--test_subset_frac", type=float, default=0.5)
    parser.add_argument("--image_train_cyc", type=int, default=10)
    parser.add_argument("--warmup_iters_pretrain", type=int, default=50)
    parser.add_argument("--lr_decay_iters_pretrain", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--use_flexible_tokens", action="store_true", default=True,
                        help="We always use classification lines + flexible scheme in training references.")
    parser.add_argument("--forced_next_count", type=int, default=10)

    parser.add_argument("--filter_findings", action="store_true", default=True)
    parser.add_argument(
        "--shuffle_findings",
        type=lambda x: str(x).lower() in {"true", "1", "yes"},
        default=True,
        help="Set to false to keep findings sentences in their original order.",
    )
    parser.add_argument(
        "--use_report_tags",
        type=lambda x: str(x).lower() in {"true", "1", "yes"},
        default=True,
        help="Include <findings>, <next>, and <impression> markers in training references.",
    )
    parser.add_argument("--max_images", type=int, default=3)
    args = parser.parse_args()
    main(args)
