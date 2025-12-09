#!/usr/bin/env python3

import os
import math
import time
import logging
import sys
from datetime import timedelta
from contextlib import nullcontext, contextmanager
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import timm
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import random
# Transformers & evaluation
from transformers import (
    GPT2Config, GPT2Tokenizer,
    BertConfig, BertModel, BertTokenizer,
    GPT2LMHeadModel, GPT2Model, GenerationConfig
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import LogitsProcessorList, LogitsProcessor
from tqdm import tqdm
import evaluate
meteor = evaluate.load("meteor")
bleu   = evaluate.load("bleu")
rouge  = evaluate.load("rouge")

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

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
################################################################################
# Logging / DDP setup
################################################################################
def setup_logger():
    timestamp = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    logfile = f"./ppo_log/train_trl_log_{timestamp}.log"
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
        ddp_rank       = int(os.environ['RANK'])
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

################################################################################
# Some dataset & utilities
################################################################################
class MIMICCXRDataset(Dataset):
    def __init__(self, csv_file, transform=None, split_mode="train",
                 filter_findings=True, max_images=3):
        import pandas as pd
        df = pd.read_csv(csv_file)
        if split_mode == "train":
            df = df[df["split"].isin(["train","validate"])]
        elif split_mode == "test":
            df = df[df["split"] == "test"]
        else:
            raise ValueError("split_mode must be 'train' or 'test'")

        if filter_findings:
            df = df[(df["has_impression"] == 1) & (df["has_findings"] == 1)]
            # df = df[(df["has_findings"] == 1)]

            # Additional filters:
            # 1) findings_tokens_gpt2 > impression_tokens_gpt2
            # 2) impression_tokens_gpt2 >= 5
            df = df[
                (df["findings_tokens_gpt2"] > df["impression_tokens_gpt2"]) &
                (df["impression_tokens_gpt2"] >= 4)
                ]

            # df = df[
            #     (df["impression_tokens_gpt2"] >= 4)
            #     ]

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
            "subject_id":      row["subject_id"],
            "study_id":        row["study_id"],
            "split":           row["split"],
            "images":          images,
            "findings":        row["findings"],
            "impression":      row["impression"],
            "last_paragraph":  row["last_paragraph"],
            "report_all":      row["report_all"],
            "chexpert_labels": [int(x) for x in row["chexpert_labels"].split(";")],
            "chexbert_labels": [int(x) for x in row["chexbert_labels"].split(";")]
        }
        return sample

def my_collate_fn(batch):
    out = {
        "subject_id":      [],
        "study_id":        [],
        "split":           [],
        "images":          [],
        "findings":        [],
        "impression":      [],
        "last_paragraph":  [],
        "report_all":      [],
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

################################################################################
# Standard text metrics (BLEU/METEOR/ROUGE)
################################################################################

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

from datasets import Dataset as HFDataset
def compute_batch_nlg_metrics(ref_texts, list_of_preds, num_proc=4) -> dict:
    ds_dict = {"reference": ref_texts, "prediction": list_of_preds}
    ds = HFDataset.from_dict(ds_dict)
    def _map_fn(example):
        out = _single_nlg_reward(example["reference"], example["prediction"])
        return out
    ds_mapped = ds.map(_map_fn, batched=False, num_proc=num_proc)
    bleus_1 = ds_mapped["BLEU_1"]
    bleus_4 = ds_mapped["BLEU_4"]
    mets  = ds_mapped["METEOR"]
    rous  = ds_mapped["ROUGE_L"]
    n = len(bleus_1)
    if n>0:
        return {
            "BLEU_1": float(np.mean(bleus_1)),
            "BLEU_4": float(np.mean(bleus_4)),
            "METEOR": float(np.mean(mets)),
            "ROUGE_L": float(np.mean(rous))
        }
    else:
        return {"BLEU_1":0.0,"BLEU_4":0.0,"METEOR":0.0,"ROUGE_L":0.0}

################################################################################
# CheXbert-based text-F1
################################################################################
class CheXbert(nn.Module):
    def __init__(self, checkpoint_path, device, p=0.1):
        super().__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        config = BertConfig.from_pretrained("bert-base-uncased")
        with torch.no_grad():
            self.bert = BertModel(config)
            self.dropout = nn.Dropout(p)
            hidden_size  = self.bert.pooler.dense.in_features
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
            cleaned.append(rep.strip().replace("\n"," "))
        tokenized = self.tokenizer(cleaned, padding='longest', truncation=True, max_length=512, return_tensors='pt')
        tokenized = {k: v.to(self.device) for k,v in tokenized.items()}
        out_bert = self.bert(**tokenized)[0]
        cls_vec  = out_bert[:,0,:]
        cls_vec  = self.dropout(cls_vec)
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
            cleaned.append(rep.strip().replace("\n"," "))
        tokenized = self.tokenizer(cleaned, padding='longest', truncation=True, max_length=512, return_tensors='pt')
        tokenized = {k:v.to(self.device) for k,v in tokenized.items()}
        out_bert = self.bert(**tokenized)[0]
        cls_vec  = out_bert[:,0,:]
        return cls_vec

class CheXbertScorer:
    def __init__(self, checkpoint_path, device='cuda'):
        self.model = CheXbert(checkpoint_path, device)

    def _to_binary(self, preds_14: torch.Tensor) -> torch.Tensor:
        B = preds_14.size(0)
        out = torch.zeros(B, 14, device=preds_14.device)
        for i in range(14):
            out[:, i] = (preds_14[:, i]==1).float()
        return out

    @torch.no_grad()
    def score_batch_f1(self, ref_texts: list, gen_texts: list, gt_labels: list, batch_size: int=64):
        """
        Return elementwise F1 for each item.
        """
        assert len(gen_texts) == len(gt_labels)
        f1_list = []
        for i in range(0, len(gen_texts), batch_size):
            gen_batch = gen_texts[i:i+batch_size]
            ref_batch = ref_texts[i:i+batch_size]

            preds_14 = self.model(gen_batch)
            pred_bin = self._to_binary(preds_14)

            gt_14 = self.model(ref_batch)
            gt_bin= self._to_binary(gt_14)

            tp = (pred_bin * gt_bin).sum(dim=1)
            fp = (pred_bin * (1 - gt_bin)).sum(dim=1)
            fn = ((1 - pred_bin) * gt_bin).sum(dim=1)
            f1_eg = tp / (tp + 0.5*(fp+fn) + 1e-8)
            f1_list.append(f1_eg)
        f1_tensor = torch.cat(f1_list, dim=0)
        return f1_tensor

    @torch.no_grad()
    def score_batch_f1_with_gt(
            self,
            gen_texts: list,
            gt_labels: list,  # ground-truth Chexbert labels from the dataset
            batch_size: int = 32
    ) -> torch.Tensor:
        """
        NEW method: compare predicted labels on 'gen_texts'
        directly with the *dataset's* ground-truth label 'gt_labels'.

        We do NOT parse or use any reference text here.
        """
        f1_list = []
        total = len(gen_texts)
        for start_i in range(0, total, batch_size):
            end_i = start_i + batch_size
            batch_gen = gen_texts[start_i:end_i]

            # Predicted 14-label (4-class each), shape [bs,14]
            preds_14 = self.model(batch_gen)
            pred_bin = self._to_binary(preds_14)

            # ground-truth = 14 (stored in dataset) => 1 means 'Positive', else 0
            # If you store them as multi-class 0..3, adapt as needed,
            # or if you store them as 0 or 1, just convert to float.
            gt_14 = torch.tensor(gt_labels[start_i:end_i], dtype=torch.long, device=self.model.device)
            # Suppose gt_14 is shape [bs,14], each is 0..3.
            # If you only want to treat label==1 as positive:
            gt_bin = (gt_14 == 1).float()

            # compute F1
            tp = (pred_bin * gt_bin).sum(dim=1)
            fp = (pred_bin * (1 - gt_bin)).sum(dim=1)
            fn = ((1 - pred_bin) * gt_bin).sum(dim=1)
            f1_eg = tp / (tp + 0.5 * (fp + fn) + 1e-8)
            f1_list.append(f1_eg)

        return torch.cat(f1_list, dim=0)  # shape=[N]

    @torch.no_grad()
    def score_batch_similarity(self, ref_texts: List[str], gen_texts: List[str]):
        batch_size = 32
        sim_list = []
        for i in range(0, len(gen_texts), batch_size):
            gen_batch = gen_texts[i:i+batch_size]
            ref_batch = ref_texts[i:i+batch_size]
            ref_vecs  = self.model.get_cls_vector(ref_batch)
            gen_vecs  = self.model.get_cls_vector(gen_batch)
            sim_list.append(F.cosine_similarity(gen_vecs, ref_vecs, dim=1))
        return torch.cat(sim_list, dim=0)

################################################################################
# SWIN
################################################################################

class SwinImageEncoder(nn.Module):
    def __init__(self, model_name, out_dim=768):
        super().__init__()
        self.swin = timm.create_model(model_name, pretrained=True, img_size=512, features_only=True)

        for n, p in self.swin.named_parameters():
            if "norm" not in n:
                p.requires_grad_(False)

        self.proj = nn.Conv2d(768, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor):
        feats_list = self.swin(x)
        feats = feats_list[-1]  # last level
        feats = feats.permute(0,3,1,2)
        feats = self.proj(feats)
        B, C, H, W = feats.shape
        feats = feats.permute(0,2,3,1).reshape(B, H*W, C)
        return feats

def build_multi_image_encoder_states(batch_images, encoder, device='cuda', if_frozen=False):
    B = len(batch_images)
    all_imgs = [img for row in batch_images for img in row]
    counts   = [len(row) for row in batch_images]
    total_images = len(all_imgs)
    if total_images==0:
        out_dim = encoder.proj.out_channels
        feats_padded = torch.zeros(B,1,out_dim,device=device)
        mask_padded  = torch.zeros(B,1,dtype=torch.long,device=device)
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

################################################################################
# GPT2 wrapper for .generate() with cross-attention
################################################################################
class GPT2GenerationWrapper(GPT2LMHeadModel):
    """
    Minimal wrapper that holds onto cross-attention states (feats_padded, mask_padded)
    so we can use huggingface .generate().
    """
    def __init__(self, base_model: GPT2LMHeadModel):
        super().__init__(base_model.config)
        self.transformer = base_model.transformer
        self.lm_head     = base_model.lm_head
        self.config      = base_model.config
        self.enc_feats   = None
        self.enc_mask    = None

    def set_encoder_states(self, feats: torch.Tensor, mask: torch.Tensor):
        self.enc_feats = feats
        self.enc_mask  = mask

    def clear_encoder_states(self):
        self.enc_feats = None
        self.enc_mask  = None

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        model_inputs = {
            "input_ids": input_ids,
            "encoder_hidden_states": self.enc_feats,
            "encoder_attention_mask": self.enc_mask
        }
        if "attention_mask" in kwargs:
            model_inputs["attention_mask"] = kwargs["attention_mask"]
        if "past_key_values" in kwargs:
            model_inputs["past_key_values"] = kwargs["past_key_values"]
        if "use_cache" in kwargs:
            model_inputs["use_cache"] = kwargs["use_cache"]
        return model_inputs

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
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state
        logits        = self.lm_head(hidden_states)
        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

################################################################################
# MultiImageActor: aggregator classification + GPT2
################################################################################
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
        # self.gradient_checkpointing_enable()

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

################################################################################
# MultiImageCritic
################################################################################
class MultiImageCritic(nn.Module):
    def __init__(self,
                 hidden_dim=768,
                 swin_name="swinv2_tiny_window16_256.ms_in1k",
                 gpt2_name="gpt2"):
        super().__init__()
        self.encoder = SwinImageEncoder(swin_name, out_dim=hidden_dim)
        config = GPT2Config.from_pretrained(gpt2_name)
        config.add_cross_attention = True
        config.use_cache = True
        # config.n_head = 12
        # config.n_embd = 768
        config.n_layer = 6
        self.gpt2       = GPT2Model(config)
        self.value_head = nn.Linear(hidden_dim, 1)
        # self.gradient_checkpointing_enable()

    def forward(self,
                batch_images,
                partial_tokens,
                device='cuda',
                if_frozen=True,
                attention_mask=None,
                position_ids=None):
        feats_padded, mask_padded = build_multi_image_encoder_states(
            batch_images, self.encoder, device=device, if_frozen=if_frozen
        )
        out = self.gpt2(
            input_ids=partial_tokens.to(device),
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=feats_padded,
            encoder_attention_mask=mask_padded,
            return_dict=True
        )
        hidden = out.last_hidden_state  # [B, seq_len, hidden_dim]
        val    = self.value_head(hidden).squeeze(-1)
        return val

################################################################################
# Utility for partials
################################################################################
def pad_partials(partials: list, pad_id: int):
    lengths = [p.size(1) for p in partials]
    max_len = max(lengths) if lengths else 1
    B = len(partials)
    device_ = partials[0].device if B>0 else "cpu"
    out  = torch.full((B,max_len), pad_id, dtype=torch.long, device=device_)
    mask = torch.zeros(B,max_len, dtype=torch.long, device=device_)
    for i, seq in enumerate(partials):
        seq_len = seq.size(1)
        # out[i,:seq_len]  = seq[0]
        out[i, :seq_len] = seq[0, :seq_len]
        mask[i,:seq_len] = 1
    return out, mask

def first_true_indices(bools: torch.Tensor, dtype=torch.long):
    row_len = bools.size(-1)
    zero_or_index = row_len*(~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values

def compute_pos_only_f1(preds_14: torch.Tensor, gold_14: torch.Tensor, eps=1e-8) -> float:
    pred_bin = (preds_14 == 1).float()
    gold_bin = (gold_14 == 1).float()
    tp = (pred_bin * gold_bin).sum(dim=1)
    fp = (pred_bin * (1 - gold_bin)).sum(dim=1)
    fn = ((1 - pred_bin) * gold_bin).sum(dim=1)
    f1_eg = tp / (tp + 0.5 * (fp + fn) + eps)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(f1_eg.mean().item()), float(precision.mean().item()), float(recall.mean().item())


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
################################################################################
# PPO pipeline: batch_generation => generation_process => advantage => ...
################################################################################


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
    max_seq_len=32,
    test_subset_frac=0.2,
    num_beams=1
):
    """
    Evaluates 'actor' by:
      - Generating FINDINGS + IMPRESSION text
      - Parsing out the FINDINGS part and IMPRESSION part
      - Computing aggregator classification F1
      - Computing NLG metrics only on the FINDINGS portion
      - Computing CheXbert-based F1 for both FINDINGS and IMPRESSION
    """

    ddp = isinstance(actor, DDP)
    base_actor = actor.module if ddp else actor

    hf_model = GPT2BeamSearchWrapper(base_actor)
    hf_model.eval()
    hf_model.to(device)

    # We'll store references for FINDINGS and IMPRESSION separately
    all_ref_findings = []
    all_ref_impressions = []

    # We'll store generated FINDINGS and IMPRESSION separately
    all_gen_findings = []
    all_gen_impressions = []

    all_gt = []

    # For aggregator classification
    all_class_preds = []
    all_class_gts = []

    # DataLoader subset logic
    total_test = len(data_loader)
    max_test_batches = int(test_subset_frac * total_test)
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

        # ---------------------------------------------------------
        # 1) Collect ground-truth references for FINDINGS & IMPRESSION
        # ---------------------------------------------------------
        # Instead of building a single "FINDINGS: ... IMPRESSION: ...",
        # we store them separately for direct comparison & NLG on FINDINGS only.
        for i in range(B):
            f_text = batch_data["findings"][i] or ""
            i_text = batch_data["impression"][i] or ""
            all_ref_findings.append(f_text)
            all_ref_impressions.append(i_text)

        all_gt.extend(batch_data["chexbert_labels"])
        # ---------------------------------------------------------
        # 2) Aggregator classification F1
        # ---------------------------------------------------------
        feats_padded, mask_padded = build_multi_image_encoder_states(
            batch_data["images"], base_actor.encoder, device=device, if_frozen=True
        )
        aggregator_out = base_actor.aggregator(
            feats_padded, src_key_padding_mask=(mask_padded == 0)
        )
        mask_float = mask_padded.float().unsqueeze(-1)
        feats_sum = (aggregator_out * mask_float).sum(dim=1)
        denom = mask_float.sum(dim=1).clamp_min(1e-8)
        feats_pooled = feats_sum / denom

        # Classification head => 14 binary logits => threshold at 0.5
        cls_logits = base_actor.cls_head(feats_pooled)  # shape [B,14]
        preds_bin = (torch.sigmoid(cls_logits) >= 0.5).long()
        all_class_preds.append(preds_bin.cpu())

        # Convert CheXbert 0..3 => binary (1 => Present, else 0)
        gt_labels_full = torch.tensor(batch_data["chexbert_labels"], dtype=torch.long)
        gt_bin = (gt_labels_full == 1).long()
        all_class_gts.append(gt_bin)

        # ---------------------------------------------------------
        # 3) Generate text (Findings + Impression) with beam search
        # ---------------------------------------------------------
        # We'll start from prefix "Findings:", then decode up to <eos>
        prefix_text = "Findings:"
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        input_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)

        generated = hf_model.generate(
            input_ids=input_ids,
            encoder_hidden_states=feats_padded,
            encoder_attention_mask=mask_padded,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_seq_len,
            do_sample=False,
            use_cache=False,
            num_beams=num_beams
        )

        # ---------------------------------------------------------
        # Parse out the FINDINGS portion and the IMPRESSION portion
        # ---------------------------------------------------------
        for i in range(generated.size(0)):
            tokens_i = generated[i, :]
            gen_text_full = tokenizer.decode(tokens_i, skip_special_tokens=True)

            # Extract the portion after "Findings:" until "Impression:" (or end)
            match_find = re.search(r"Findings:\s*(.*?)(?:Impression:|$)",
                                   gen_text_full, re.IGNORECASE)
            if match_find:
                only_find = match_find.group(1).strip()
            else:
                only_find = gen_text_full.strip()  # fallback

            # Extract the portion after "Impression:" if present
            match_impr = re.search(r"Impression:\s*(.*)", gen_text_full, re.IGNORECASE)
            if match_impr:
                only_impr = match_impr.group(1).strip()
            else:
                only_impr = ""

            all_gen_findings.append(only_find)
            all_gen_impressions.append(only_impr)

    # =====================================================================
    # Compute final metrics
    # =====================================================================
    # 1) Classification aggregator F1
    if len(all_class_preds) == 0:
        # No data processed => return zeros
        return 0.0, 0.0, 0.0, 0.0, 0.0

    all_class_preds_tensor = torch.cat(all_class_preds, dim=0)
    all_class_gts_tensor = torch.cat(all_class_gts, dim=0)
    classification_f1, p_, r_ = compute_pos_only_f1(
        all_class_preds_tensor, all_class_gts_tensor
    )

    # 2) NLG metrics => only on FINDINGS
    #    references: all_ref_findings, generations: all_gen_findings
    nlg_dict = compute_batch_nlg_metrics(all_ref_findings, all_gen_findings, num_proc=4)

    # 3) F1 (Findings) => compare ground-truth findings vs. generated findings
    f1_vals_find = chex_scorer.score_batch_f1(all_ref_findings, all_gen_findings, all_gt)
    f1_findings = float(f1_vals_find.mean().item())

    # 4) F1 (Impression) => compare ground-truth impressions vs. generated impressions
    f1_vals_impr = chex_scorer.score_batch_f1(all_ref_impressions, all_gen_impressions, all_gt)
    f1_impression = float(f1_vals_impr.mean().item())

    # ---------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------
    if ddp_rank0:
        logging.info(f"[Eval => aggregator Classification F1={classification_f1:.3f}, "
                     f"P={p_:.3f}, R={r_:.3f}]")
        logging.info(f"[Eval => Findings-based NLG] BLEU1={nlg_dict['BLEU_1']:.3f}, "
                     f"BLEU4={nlg_dict['BLEU_4']:.3f}, METEOR={nlg_dict['METEOR']:.3f}, "
                     f"ROUGE_L={nlg_dict['ROUGE_L']:.3f}")
        logging.info(f"[Eval => CheX-F1] Findings={f1_findings:.3f} | Impression={f1_impression:.3f}")

        # Optionally show some examples
        for i in range(min(2, len(all_gen_findings))):
            logging.info(f"--- Example {i} ---")
            logging.info(f">> REF-FIND: {all_ref_findings[i]}")
            logging.info(f">> GEN-FIND: {all_gen_findings[i]}")
            logging.info(f">> REF-IMPR: {all_ref_impressions[i]}")
            logging.info(f">> GEN-IMPR: {all_gen_impressions[i]}")

    # Return whatever metrics you need. Adjust as necessary.
    # For example: (classification_f1, BLEU1, BLEU4, METEOR, ROUGE_L, f1_findings, f1_impression)
    return (
        classification_f1,
        nlg_dict["BLEU_1"],
        nlg_dict["BLEU_4"],
        nlg_dict["METEOR"],
        nlg_dict["ROUGE_L"],
        f1_findings,
        f1_impression
    )

# ------------------------------------------------------------
# critic-reranked, two-stage sampling evaluation
# ------------------------------------------------------------


@torch.no_grad()
def evaluate_model_flexible_next(
    actor:       nn.Module,
    data_loader: DataLoader,
    tokenizer:   GPT2Tokenizer,
    device:      torch.device,
    chex_scorer,
    *,
    forced_next_count: int = 0,        # ≥ 0; 0 ⇒ vanilla decoding
    max_seq_len:       int = 100,
    test_subset_frac:  float = 0.2,
):
    ddp        = isinstance(actor, DDP)
    base_actor = actor.module if ddp else actor
    base_actor.eval().to(device)

    # Identifiers and tokens -----------------------------------------------
    eos_id  = tokenizer.eos_token_id or tokenizer.bos_token_id
    next_id = (tokenizer.convert_tokens_to_ids("<next>")
               if "<next>" in tokenizer.get_vocab() else None)
    impr_id = (tokenizer.convert_tokens_to_ids("<impression>")
               if "<impression>" in tokenizer.get_vocab() else None)

    if forced_next_count and next_id is None:
        raise ValueError("forced_next_count > 0 but <next> token not in vocab")

    prefix_ids = torch.tensor(
        tokenizer.encode("Findings:", add_special_tokens=False),
        device=device
    )

    # Storage for references and generations -------------------------------
    # For aggregator classification
    cls_pred, cls_true = [], []
    all_gt = []

    # For textual references/generations
    ref_findings,   gen_findings    = [], []
    ref_impressions, gen_impressions = [], []

    max_batches = int(len(data_loader) * test_subset_frac)
    iterator = tqdm(enumerate(data_loader), total=max_batches,
                    desc=f"Eval-Next-Forcing{forced_next_count}",
                    disable= ddp and int(os.environ.get("RANK", 0)) != 0)

    # ======================================================================
    for b_idx, batch in iterator:
        if b_idx >= max_batches:
            break
        B = len(batch["report_all"])
        if B == 0:
            continue

        # 1) Collect reference findings/impressions
        for fnd, imp in zip(batch["findings"], batch["impression"]):
            ref_findings.append(fnd)
            ref_impressions.append(imp)

        all_gt.extend(batch["chexbert_labels"])

        # 2) Aggregator classification F1
        feats, mask = build_multi_image_encoder_states(
            batch["images"], base_actor.encoder, device=device, if_frozen=True
        )
        pooled = (base_actor.aggregator(feats, src_key_padding_mask=(mask == 0))
                  * mask.float().unsqueeze(-1)).sum(1) \
                 / mask.sum(1, keepdim=True).clamp_min(1e-8)
        pred_bin = (torch.sigmoid(base_actor.cls_head(pooled)) >= 0.5).long()
        cls_pred.append(pred_bin.cpu())
        cls_true.append((torch.tensor(batch["chexbert_labels"]) == 1).long())

        # ==================================================================
        #            1) FINDINGS Generation
        # ==================================================================
        parts  = [prefix_ids.unsqueeze(0).clone() for _ in range(B)]
        n_next = [0]*B
        done   = [False]*B

        for _ in range(max_seq_len):
            active = [i for i,d in enumerate(done) if not d]
            if not active:
                break

            toks, _ = pad_partials([parts[i] for i in active],
                                   tokenizer.pad_token_id)
            imgs    = [batch["images"][i] for i in active]
            logits  = base_actor(imgs, toks, device=device, do_sample=False)
            nxt     = logits[:, -1].argmax(-1)  # [len(active)]

            for ai, bi in enumerate(active):
                tok = nxt[ai].item()

                if n_next[bi] < forced_next_count:
                    # Force <next> token until we meet forced_next_count
                    if (tok == eos_id or (impr_id and tok == impr_id)) and (next_id is not None):
                        tok = next_id
                    if tok == next_id:
                        n_next[bi] += 1
                else:
                    # Once forced_next_count reached, next token => <impression> or <eos>
                    if not (tok == eos_id or (impr_id and tok == impr_id)):
                        tok = impr_id if impr_id is not None else eos_id
                    done[bi] = True

                parts[bi] = torch.cat([parts[bi], toks.new_tensor([[tok]])], 1)
                if tok == eos_id or (impr_id and tok == impr_id):
                    done[bi] = True

        # Decode to get generated "Findings"
        dec_findings = [tokenizer.decode(p.squeeze(0), skip_special_tokens=True)
                        for p in parts]
        gen_findings.extend(dec_findings)

        # ==================================================================
        #            2) IMPRESSION Generation
        # ==================================================================
        for i in range(B):
            prefix_text = dec_findings[i].strip()
            if impr_id is not None and "<impression>" not in prefix_text.lower():
                prefix_text += " <impression>"

            seq = torch.tensor(tokenizer.encode(prefix_text, add_special_tokens=False),
                               device=device).unsqueeze(0)
            for _ in range(max_seq_len):
                logits = base_actor([batch["images"][i]], seq,
                                    device=device, do_sample=False)
                nx = logits[:, -1].argmax(-1).item()
                seq = torch.cat([seq, seq.new_tensor([[nx]])], 1)
                if nx == eos_id:
                    break

            out_str = tokenizer.decode(seq.squeeze(0), skip_special_tokens=True)
            # parse out impression text
            m = re.search(r"[Ii]mpression:\s*(.*)", out_str)
            gen_imp = m.group(1).strip() if m else ""
            gen_impressions.append(gen_imp)

    # ======================================================================
    #  Compute Metrics
    # ======================================================================
    # 1) Classification (aggregator) F1
    cls_pred_cat = torch.cat(cls_pred)
    cls_true_cat = torch.cat(cls_true)
    cls_f1, cls_P, cls_R = compute_pos_only_f1(cls_pred_cat, cls_true_cat)

    # 2) NLG metrics on FINDINGS only
    #    reference = ref_findings, generation = gen_findings
    nlg = compute_batch_nlg_metrics(ref_findings, gen_findings, num_proc=4)

    # 3) F1 for FINDINGS → compare model's generated findings vs. ground-truth findings
    f1_findings = chex_scorer.score_batch_f1(ref_findings, gen_findings, all_gt).mean().item()

    # 4) F1 for IMPRESSION → compare model's generated impression vs. ground-truth impression
    f1_impression = chex_scorer.score_batch_f1(ref_impressions, gen_impressions, all_gt).mean().item()

    # Logging
    logging.info(f"[FlexNext] Class-F1={cls_f1:.3f}  P={cls_P:.3f}  R={cls_R:.3f}")
    logging.info(
        f"[FlexNext] (Findings NLG) BLEU1={nlg['BLEU_1']:.3f}  BLEU4={nlg['BLEU_4']:.3f}  "
        f"METEOR={nlg['METEOR']:.3f}  ROUGE_L={nlg['ROUGE_L']:.3f}"
    )
    logging.info(f"[FlexNext] Findings-F1={f1_findings:.3f}  Impression-F1={f1_impression:.3f}")

    # Optionally log a few examples
    for i in range(min(2, len(gen_findings))):
        logging.info(f"[FlexNext Sample {i}]  GEN-FIND: {gen_findings[i]}")
        logging.info(f"[FlexNext Sample {i}]  REF-FIND: {ref_findings[i]}")
    for i in range(min(2, len(gen_impressions))):
        logging.info(f"[FlexNext Sample {i}]  GEN-IMPR: {gen_impressions[i]}")
        logging.info(f"[FlexNext Sample {i}]  REF-IMPR: {ref_impressions[i]}")

    # Return whatever metrics you need
    return cls_f1, nlg["BLEU_1"], nlg["BLEU_4"], nlg["METEOR"], nlg["ROUGE_L"], f1_findings, f1_impression


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
    """Compute mean of tensor with masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when mini_batch_size=1;"
                "try increase the mini_batch_size or gradient_accumulation_steps"
            )
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance

def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def pad(tensors: list[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    output_shape = np.max([t.shape for t in tensors], 0).tolist()
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)
    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t
    return output

def get_reference_text(sample: dict) -> str:
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

    chain = []
    chain.append(f"Findings: ")
    if len(find_sents) == 0:
        chain.append(f""
                     f"{findings_val}")
    else:
        for idx_, fsent in enumerate(find_sents, 1):
            chain.append(f"{fsent}.")
            if idx_ < len(find_sents):
                chain.append("<next>")
    if impression_val:
        chain.append(f"<impression> Impression: {impression_val}")

    flexible_text = " ".join(chain)
    final_ref = flexible_text
    return final_ref

@torch.no_grad()
def batch_generation(
    actor,
    batch_data,
    tokenizer,
    device,
    max_new_tokens,
    temperature,
):
    actor.eval()
    B = len(batch_data["report_all"])
    ref_texts = []
    for i in range(B):
        samp_i = {
            "findings": batch_data["findings"][i],
            "impression": batch_data["impression"][i],
            "report_all": batch_data["report_all"][i]
        }
        # We do not specifically insert <next>/<impression> in the prefix text here,
        # just store the reference for reward logging.
        ref_texts.append(get_reference_text(samp_i))  # CHANGED: calls the same function but not forced usage of tokens

    base_actor = actor.module if isinstance(actor, DDP) else actor
    gen_wrapper = GPT2GenerationWrapper(base_actor).to(device)

    feats_padded, feats_mask = build_multi_image_encoder_states(
        batch_data["images"], base_actor.encoder, device=device, if_frozen=True
    )
    gen_wrapper.set_encoder_states(feats_padded, feats_mask)

    prefix_text = "Findings:"
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    prefix_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    prefix_batch = prefix_ids.unsqueeze(0).expand(B, -1)
    context_length = prefix_batch.shape[1]

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        top_k=0,
        output_scores=True,
        return_dict_in_generate=True
    )

    outputs = gen_wrapper.generate(
        input_ids=prefix_batch,
        generation_config=generation_config,
        use_cache=False
    )
    gen_wrapper.clear_encoder_states()
    out_batch = outputs.sequences
    logits = torch.stack(outputs.scores, 1)
    query_responses = out_batch
    responses = out_batch[:, context_length:]

    return query_responses, responses, logits.cpu(), ref_texts

@torch.no_grad()
def generation_process(
    ref_actor,
    critic,
    tokenizer,
    device,
    chex_scorer,
    responses,
    query_responses,
    images,
    ref_texts,
    chexbert_labels,
    temperature,
):
    ref_actor.eval()
    critic.eval()
    B = responses.shape[0]
    prefix_text = "Findings:"
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    prefix_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    context_length = len(prefix_ids)

    final_texts = []
    lens = []
    for i in range(B):
        row_tokens = responses[i]
        pad_positions = (row_tokens == tokenizer.pad_token_id).nonzero(as_tuple=True)
        if len(pad_positions[0]) == 0:
            length_i = row_tokens.size(0)
        else:
            length_i = pad_positions[0][0].item()
        gen_text = tokenizer.decode(row_tokens[:length_i], skip_special_tokens=True)
        final_texts.append(gen_text)
        lens.append(length_i)

    def extract_impression_text(full_generation: str) -> str:
        marker = "impression:"
        lower_text = full_generation.lower()
        idx = lower_text.find(marker)
        if idx != -1:
            # cut from the end of '<impression>'
            return full_generation[idx + len(marker):].strip()
        # fallback if marker not found
        return full_generation.strip()

    def extract_findings_text(generated_str: str) -> str:
        """
        Extract the text under 'FINDINGS:' up to either 'IMPRESSION:' or the end of the string.
        Adjust the regex or logic as needed for your model’s format.
        """
        # Example: "FINDINGS: Lungs are clear. IMPRESSION: No acute abnormality."
        pattern = re.compile(r"FINDINGS:\s*(.*?)(?=IMPRESSION:|$)", re.IGNORECASE | re.DOTALL)
        match = pattern.search(generated_str)
        if match:
            return match.group(1).strip()
        else:
            return generated_str.strip()  # fallback if no "FINDINGS:" found

    impression_texts = []
    for gen_str in final_texts:
        impression_part = extract_impression_text(gen_str)
        impression_texts.append(impression_part)

    ref_impressions = []
    for ref_str in ref_texts:
        ref_part = extract_impression_text(ref_str)
        ref_impressions.append(ref_part)

    gen_findings_texts = []
    for gen_str in final_texts:
        gen_findings_texts.append(extract_findings_text(gen_str))

    final_rewards = chex_scorer.score_batch_f1_with_gt(impression_texts, chexbert_labels)

    scores_np = final_rewards.cpu().numpy()

    threshold = 0.5
    good_mask = (final_rewards >= threshold)  # bool
    bad_mask = ~good_mask
    good_idx = good_mask.nonzero(as_tuple=True)[0]  # shape[..]
    bad_idx = bad_mask.nonzero(as_tuple=True)[0]

    # gather each subset's scores for sorting
    good_scores = scores_np[good_idx.cpu().numpy()]
    bad_scores = scores_np[bad_idx.cpu().numpy()]

    # Sort good_idx by descending score => keep top items if we have more than needed
    sorted_good_idx = good_idx[torch.argsort(torch.tensor(good_scores, device=device), descending=True)]
    # Sort bad_idx by ascending score => keep the "worst" items if we have too many
    sorted_bad_idx = bad_idx[torch.argsort(torch.tensor(bad_scores, device=device), descending=False)]

    len_good = sorted_good_idx.size(0)
    len_bad = sorted_bad_idx.size(0)
    print('good/bad ratio: ', str(len_good / len_bad))
    ref_logprobs_ls = []
    batch_size = 2
    for i in range(0, B, batch_size):
        image = images[i: i + batch_size]
        query_response = query_responses[i: i + batch_size]
        response = responses[i: i + batch_size]
        attention_mask = (query_response != tokenizer.pad_token_id).long()
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        with torch.no_grad():
            ref_logit = ref_actor(
                image,
                query_response,
                device=device,
                if_frozen=True,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            ref_logit /= temperature + 1e-7
            ref_log_soft = F.log_softmax(ref_logit[:, context_length - 1: -1], dim=-1)
            ref_logprob = torch.gather(ref_log_soft, 2, response.unsqueeze(-1)).squeeze(-1)
            ref_logprobs_ls.append(ref_logprob)
    ref_logprobs = torch.cat(ref_logprobs_ls, 0)

    values_ls = []
    for i in range(0, B, batch_size):
        image = images[i: i + batch_size]
        query_response = query_responses[i: i + batch_size]
        attention_mask = (query_response != tokenizer.pad_token_id).long()
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        with torch.no_grad():
            val_critic = critic(
                image,
                query_response,
                device=device,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            values_ls.append(val_critic)
    values_temp = torch.cat(values_ls, 0)
    values = values_temp[:, context_length - 1 : -1]

    sequence_lengths = first_true_indices(responses == tokenizer.pad_token_id) - 1
    scores = torch.tensor(final_rewards).to(device)
    return ref_logprobs, sequence_lengths, scores, values

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
    last_epoch = ckpt.get("last_epoch", ckpt.get("ppo_iter", 0))
    logging.info(f"... resume done. last_epoch={last_epoch}")
    return last_epoch


def main(args):
    ddp, device, master_process, ddp_world_size = setup_distributed(args)
    if master_process:
        logger, logfile = setup_logger()
        logger.info("Starting the training script...")
        os.makedirs(args.ckpt_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # ----------------------------------------------------------------------------
    # CHANGED: Add <next> and <impression> to the tokenizer if not already present
    # ----------------------------------------------------------------------------
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Add classification-related tokens if they're not already in the vocab
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
    if "<next>" not in tokenizer.get_vocab():
        additional_specials.append("<next>")
    if "<impression>" not in tokenizer.get_vocab():
        additional_specials.append("<impression>")

    all_new_specials = chex_condition_specials + chex_class_specials + additional_specials
    if len(all_new_specials) > 0:
        tokenizer.add_special_tokens({"additional_special_tokens": all_new_specials})

    # Ensure we have a pad token
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    transform = T.Compose([
        T.Resize(args.image_size + 64),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    ptdtype = {'float32': torch.float32,
               'float16': torch.float16,
               'bfloat16': torch.bfloat16}[args.dtype]

    train_dataset = MIMICCXRDataset(args.csv_path, transform=transform, split_mode="train",
                                    filter_findings=args.filter_findings, max_images=args.max_images)
    test_dataset = MIMICCXRDataset(args.csv_path, transform=transform, split_mode="test",
                                   filter_findings=args.filter_findings, max_images=20)

    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler_train = DistributedSampler(
            train_dataset,
            shuffle=True,
            num_replicas=ddp_world_size,
            rank=int(os.environ['RANK']),
            drop_last=True
        )
    else:
        sampler_train = None

    train_rollout_loader = DataLoader(
        train_dataset,
        batch_size=args.rollout_batch_size,
        shuffle=(sampler_train is None),
        sampler=sampler_train,
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

    actor = MultiImageActor(vocab_size=len(tokenizer))
    critic = MultiImageCritic()
    critic.gpt2.resize_token_embeddings(len(tokenizer))
    logging.info(
        f"[Param Counts] actor total={sum(p.numel() for p in actor.parameters() if p.requires_grad) / 1e6:.2f}M")
    logging.info(
        f"[Param Counts] actor swin={sum(p.numel() for p in actor.encoder.parameters() if p.requires_grad) / 1e6:.2f}M")
    logging.info(
        f"[Param Counts] actor aggregator={sum(p.numel() for p in actor.aggregator.parameters() if p.requires_grad) / 1e6:.2f}M")
    logging.info(
        f"[Param Counts] actor gpt={sum(p.numel() for p in actor.transformer.parameters() if p.requires_grad) / 1e6:.2f}M")

    ckpt = torch.load('/home/shyoon/rrg_ttc/forced_generation/final_checkpoints/actor_warmup_cycle_68_larger_epoch_1.pth', map_location=device)
    if "actor_state" in ckpt:
        state_dict = ckpt["actor_state"]
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        actor.load_state_dict(state_dict, strict=True)

    critic.encoder.load_state_dict(actor.encoder.state_dict(), strict=True)

    if master_process:
        logging.info(f"critic={sum(p.numel() for p in critic.parameters() if p.requires_grad) / 1e6:.2f}M")

    actor.to(device)
    critic.to(device)

    ref_actor = MultiImageActor(vocab_size=len(tokenizer))
    ref_actor.load_state_dict(actor.state_dict(), strict=True)
    ref_actor.to(device)
    ref_actor.eval()
    for p in ref_actor.parameters():
        p.requires_grad = False

    actor_optim = optim.AdamW(actor.parameters(), lr=args.ppo_actor_lr, betas=(0.9, 0.95))
    critic_optim = optim.AdamW(critic.parameters(), lr=args.ppo_critic_lr, betas=(0.9, 0.95))

    start_ppo_iter = 0
    if args.resume_ppo_path and os.path.isfile(args.resume_ppo_path):
        ckpt_ppo = torch.load(args.resume_ppo_path, map_location=device)
        actor_state_dict = ckpt_ppo.get("actor_state", {})
        critic_state_dict = ckpt_ppo.get("critic_state", {})
        unwanted_prefix = '_orig_mod.'
        for k in list(actor_state_dict.keys()):
            if k.startswith(unwanted_prefix):
                new_k = k[len(unwanted_prefix):]
                actor_state_dict[new_k] = actor_state_dict.pop(k)
        for k in list(critic_state_dict.keys()):
            if k.startswith(unwanted_prefix):
                new_k = k[len(unwanted_prefix):]
                critic_state_dict[new_k] = critic_state_dict.pop(k)
        actor.load_state_dict(actor_state_dict, strict=True)
        critic.load_state_dict(critic_state_dict, strict=True)
        actor_optim.load_state_dict(ckpt_ppo.get("actor_optim", {}))
        critic_optim.load_state_dict(ckpt_ppo.get("critic_optim", {}))
        start_ppo_iter = ckpt_ppo.get("ppo_iter", 0)
        if master_process:
            logging.info(f"[Resume PPO] from iter={start_ppo_iter}")

    if args.compile_model:
        actor = torch.compile(actor)
        critic = torch.compile(critic)
        ref_actor = torch.compile(ref_actor)

    if ddp:
        actor = DDP(actor, device_ids=[int(os.environ['LOCAL_RANK'])], find_unused_parameters=True)
        critic = DDP(critic, device_ids=[int(os.environ['LOCAL_RANK'])], find_unused_parameters=True)

    chex_scorer = CheXbertScorer(args.chexbert_path, device=device)
    outer_iter = start_ppo_iter
    if args.eval_only:
        if master_process:
            classification_f1, BLEU1, BLEU4, METEOR, ROUGE_L, f1_findings, f1_impression = evaluate_model(
                actor if not ddp else actor.module,
                test_loader,
                tokenizer,
                device,
                chex_scorer,
                max_seq_len=args.max_seq_len,
                test_subset_frac=args.test_subset_frac,
                num_beams=1,
            )
            cls_f1, b1, bleu4, meteor, rouge, f1_findings, impr_f1 = evaluate_model_flexible_next(
                actor, test_loader, tokenizer, device, chex_scorer,
                forced_next_count=12,  # e.g. require at least 2 <next>
                max_seq_len=args.max_seq_len,
                test_subset_frac=args.test_subset_frac,
            )

    else:
        # Main PPO training loop
        num_step = 0
        running_mean = torch.zeros((), device=device)
        running_var = torch.ones((), device=device)
        momentum = 0.01
        kl_coef = args.kl_coef
        # for pg in actor_optim.param_groups:
        #     pg["lr"] = 1e-5
        # for pg in critic_optim.param_groups:
        #     pg["lr"] = 1e-5
        if master_process:
            logging.info(f"[PPO Iteration {start_ppo_iter + 1}] => LR={args.ppo_actor_lr:.6g}")
        for it in range(start_ppo_iter, args.ppo_iters):
            if ddp and hasattr(train_rollout_loader.sampler, 'set_epoch'):
                train_rollout_loader.sampler.set_epoch(it)
            query_responses = []
            prefix_lengths = []
            responses = []
            images = []
            logits = []
            ref_texts = []
            chexbert_labels = []

            total_train = len(train_rollout_loader)
            target_examples_global = 256  # <- what you asked for
            world_size = torch.distributed.get_world_size() if ddp else 1

            examples_per_gpu = math.ceil(target_examples_global / world_size)
            max_batches = math.ceil(examples_per_gpu / args.rollout_batch_size)
            prefix_text = "Findings:"
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            prefix_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device)
            if master_process:
                iterator = tqdm(
                    enumerate(train_rollout_loader),
                    total=min(max_batches, total_train),
                    desc=f"PPO Iter {it + 1} Rollouts"
                )
            else:
                iterator = enumerate(train_rollout_loader)

            N = 4
            for b_i, batch_data in iterator:
                if b_i >= max_batches:
                    break
                B = len(batch_data["report_all"])
                if B == 0:
                    continue
                torch.cuda.empty_cache()

                for _ in range(N):
                    with torch.no_grad():
                        query_response, response, logit, ref_text = batch_generation(
                            actor,
                            batch_data,
                            tokenizer,
                            device,
                            max_new_tokens=args.max_seq_len,
                            temperature=args.temperature
                        )
                    query_responses.append(query_response)
                    responses.append(response)
                    logits.append(logit)
                    ref_texts.extend(ref_text)
                    images.extend(batch_data["images"])
                    chexbert_labels.extend(batch_data["chexbert_labels"])

            lengths = []
            for batch_resp in responses:
                for resp_tensor in batch_resp:
                    lengths.append(resp_tensor.size(0))
            if lengths:
                mean_len = sum(lengths) / len(lengths)
                logging.info(f"[PPO iter {it + 1}] => Mean response length: {mean_len:.2f}")

            padded_query_responses = pad(query_responses, padding_value=tokenizer.pad_token_id, padding_side="right")
            padded_responses = pad(responses, padding_value=tokenizer.pad_token_id, padding_side="right")
            padded_logitss = pad(logits, padding_value=0, padding_side="right")

            batch_size = padded_query_responses.shape[0] * padded_query_responses.shape[1]
            padded_responses = padded_responses.view(-1, padded_responses.shape[-1])[:batch_size]
            padded_query_responses = padded_query_responses.view(-1, padded_query_responses.shape[-1])[:batch_size]
            padded_logitss = padded_logitss.view(-1, *padded_logitss.shape[2:])[:batch_size]
            logprobs = F.log_softmax(padded_logitss, dim=-1)
            logprobs = torch.gather(logprobs, 2, padded_responses.to('cpu').unsqueeze(-1)).squeeze(-1).to(device)
            del padded_logitss, logits, logit
            torch.cuda.empty_cache()

            ref_logprobs, sequence_lengths, scores, values = generation_process(
                ref_actor,
                critic,
                tokenizer,
                device,
                chex_scorer,
                padded_responses,
                padded_query_responses,
                images,
                ref_texts,
                chexbert_labels,
                args.temperature,
            )

            logging.info(
                f"[PPO iter {it + 1}] => Reward Mean,Min,Max={scores.mean():.3f}, {scores.min():.3f}, {scores.max():.3f}")

            query_responses = padded_query_responses
            responses = padded_responses

            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
            logprobs = torch.masked_fill(logprobs, padding_mask, 1.0)
            ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, 1.0)
            sequence_lengths_p1 = sequence_lengths + 1
            padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
            values = torch.masked_fill(values, padding_mask_p1, 0)
            kl = logprobs - ref_logprobs
            non_score_reward = - kl_coef * kl
            rewards = non_score_reward.clone()
            actual_start = torch.arange(rewards.size(0), device=rewards.device)
            actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
            rewards[[actual_start, actual_end]] += scores

            lastgaelam = 0
            gamma = 1
            lam = 0.95
            advantages_reversed = []
            gen_length = responses.shape[1]
            for t in reversed(range(gen_length)):
                nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = rewards[:, t] + gamma * nextvalues - values[:, t]
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], axis=1)
            returns = advantages + values
            advantages = masked_whiten(advantages, ~padding_mask)
            advantages = torch.masked_fill(advantages, padding_mask, 0)

            context_length = len(prefix_ids)
            total_ex = advantages.shape[0]  # total examples left

            per_gpu_eff_bs = args.effective_batch_size / (ddp_world_size if ddp else 1)
            grad_accum = math.ceil(per_gpu_eff_bs / args.ppo_batch_size)
            big_batch_size = args.ppo_batch_size * grad_accum
            epoch_pg_losses = []
            epoch_vf_losses = []
            epoch_ratios = []
            for ppo_epoch_idx in range(args.ppo_epochs):
                b_inds = np.random.permutation(total_ex)

                for big_start in range(0, total_ex, big_batch_size):
                    big_end = big_start + big_batch_size
                    big_inds = b_inds[big_start:big_end]

                    actor_optim.zero_grad()
                    critic_optim.zero_grad()

                    # Now accumulate over micro-batches
                    for micro_start in range(big_start, big_end, args.ppo_batch_size):
                        micro_end = micro_start + args.ppo_batch_size
                        mb_inds = big_inds[micro_start - big_start:micro_end - big_start]
                        if len(mb_inds) == 0:
                            break                        
                        mb_advantage = advantages[mb_inds]
                        mb_responses = responses[mb_inds]
                        mb_query_responses = query_responses[mb_inds]
                        mb_logprobs = logprobs[mb_inds]
                        mb_return = returns[mb_inds]
                        mb_values = values[mb_inds]
                        mb_images = [images[i_b] for i_b in mb_inds]

                        attention_mask = (mb_query_responses != tokenizer.pad_token_id).long()
                        position_ids = attention_mask.cumsum(1) - attention_mask.long()

                        last_micro = (micro_end - big_start) >= len(big_inds)
                        ddp_sync_actor  = nullcontext()
                        ddp_sync_critic = nullcontext()
                        if isinstance(actor, DDP) and isinstance(critic, DDP) and not last_micro:
                            ddp_sync_actor  = actor.no_sync()
                            ddp_sync_critic = critic.no_sync()
                        # with (torch.amp.autocast('cuda', dtype=ptdtype)
                        with ddp_sync_actor, ddp_sync_critic, (torch.amp.autocast('cuda', dtype=ptdtype) if device != 'cpu' else nullcontext()):
                            logits_temp = actor(
                                mb_images,
                                mb_query_responses,
                                device=device,
                                do_sample=False,
                                if_frozen=True,
                                attention_mask=attention_mask,
                                position_ids=position_ids
                            )
                            vpred_temp = critic(
                                mb_images,
                                mb_query_responses,
                                device=device,
                                if_frozen=True,
                                attention_mask=attention_mask,
                                position_ids=position_ids
                            )

                            logits = logits_temp[:, context_length - 1: -1]
                            logits /= args.temperature + 1e-7
                            new_all_logprobs = F.log_softmax(logits, dim=-1)
                            new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)

                            pm = response_idxs[:len(mb_inds)] > (sequence_lengths[mb_inds].unsqueeze(1))
                            new_logprobs = torch.masked_fill(
                                new_logprobs, pm, 1.0
                            )
                            vpred = vpred_temp[:, context_length - 1: -1].squeeze(-1)
                            pm1 = response_idxs[:len(mb_inds)] > (sequence_lengths_p1[mb_inds].unsqueeze(1))
                            vpred = torch.masked_fill(vpred, pm1, 0)
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.clip_epsilon,
                                mb_values + args.clip_epsilon,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(vf_loss_max, ~pm1)
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)

                            pg_losses = -mb_advantage * ratio
                            clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon)
                            pg_losses2 = -mb_advantage * clipped_ratio
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, ~pm)
                            # scale the total loss by ppo_grad_accum so final gradient is stable
                            loss = (pg_loss + 0.1 * vf_loss) / grad_accum

                        loss.backward()
                        epoch_pg_losses.append(pg_loss.item())
                        epoch_vf_losses.append(vf_loss.item())
                        epoch_ratios.append(ratio.mean().item())
                        # print('ratio: ', ratio.mean().item(), 'pg_loss: ', pg_loss.item(), ' vf_loss: ', vf_loss.item())

                    # After micro-batches => step once
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    actor_optim.step()
                    critic_optim.step()
                    num_step += 1

            if master_process:
                mean_pg_loss = np.mean(epoch_pg_losses)
                mean_vf_loss = np.mean(epoch_vf_losses)
                mean_ratio = np.mean(epoch_ratios)
                logger.info(
                    f"[PPO Iter {it + 1} END] => PG Loss={mean_pg_loss:.4f}, "
                    f"Value Loss={mean_vf_loss:.4f}, ratio={mean_ratio:.4f}"
                )

            if master_process and ((it + 1) % args.eval_every_ppo == 0):
                # CHANGED: pass use_flexible_tokens if we want references to have <next>/<impression>.
                cls_f1, b1, b4, mr, rg, f1_find, f1_impr = evaluate_model(
                    actor if not ddp else actor.module,
                    test_loader,
                    tokenizer,
                    device,
                    chex_scorer,
                    max_seq_len=args.max_seq_len,
                    test_subset_frac=args.test_subset_frac,
                )
                # logging.info(f"[PPO iter {it + 1}] => Image F1={cls_f1:.3f}, BLEU4={b1:.3f}, BLEU4={b4:.3f}, METEOR={mr:.3f}, ROUGE_L={rg:.3f}, F1={txt_f1:.3f}")

            if master_process and ((it + 1) % args.save_every_ppo == 0):
                raw_actor = actor.module if ddp else actor
                raw_critic = critic.module if ddp else critic
                cdict = {
                    "actor_state": raw_actor.state_dict(),
                    "critic_state": raw_critic.state_dict(),
                    "actor_optim": actor_optim.state_dict(),
                    "critic_optim": critic_optim.state_dict(),
                    "ppo_iter": it + 1
                }
                # ckpt_name = f"./ckpt_scratch_rl/ppo_reward_f1_trl_iter_{it + 1}.pth"
                ckpt_name = f"{args.ckpt_dir}/ppo_reward_f1_trl_iter_{it + 1}.pth"
                torch.save(cdict, ckpt_name)
                logging.info(f"[Checkpoint] PPO iter {it + 1} => {ckpt_name}")

            outer_iter += 1
            print('step performed so far: ', str(num_step))

    if ddp:
        cleanup_distributed()
    if master_process:
        logging.info("All training done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="PREPROCESSED DATA FILE PATH")
    parser.add_argument("--chexbert_path", type=str, default="CHEXBERT CHECKPINT PATH")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--allow_tf32", action="store_true", default=True)
    parser.add_argument("--compile_model", action="store_true", default=False)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--pretrain_cycles", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--test_subset_frac", type=float, default=0.3)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--ppo_iters", type=int, default=1500)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ppo_subset_frac", type=float, default=0.002)
    parser.add_argument("--ppo_actor_lr", type=float, default=1e-5)
    parser.add_argument("--ppo_critic_lr", type=float, default=1e-5)

    # rollout config
    parser.add_argument("--rollout_batch_size", type=int, default=8)

    # ppo grad accumulation config
    parser.add_argument("--ppo_batch_size", type=int, default=4)
    parser.add_argument("--effective_batch_size", type=int, default=256)
    parser.add_argument("--resume_ppo_path", type=str, default='')

    # Checkpoint save options
    parser.add_argument("--eval_every_ppo", type=int, default=25)
    parser.add_argument("--save_every_ppo", type=int, default=25)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoint")
    parser.add_argument("--filter_findings", action="store_true", default=True)
    parser.add_argument("--max_images", type=int, default=3)

    args = parser.parse_args()
    main(args)
