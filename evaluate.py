#!/usr/bin/env python3
# eval_only.py
# -------------------------------------------------------------
#  Multi-image report generation 모델 평가 전용 스크립트
#  (one_shot / energy_based / cfgl / vfgs)
# -------------------------------------------------------------
import os
import sys
import re
import time
import json
import argparse
import logging
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import timedelta
from typing import List
from contextlib import nullcontext, contextmanager
from collections import defaultdict, OrderedDict
from PIL import Image

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import torchvision.transforms as T
import timm

from transformers import (
    GPT2Config, GPT2Tokenizer, GPT2LMHeadModel,
    BertConfig, BertModel, BertTokenizer,
    LogitsProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

# ---------------------------- 평가 메트릭 -----------------------------
from pycocoevalcap.bleu.bleu     import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge   import Rouge
from pycocoevalcap.cider.cider   import Cider
import yaml

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _chex_bin(tensor_or_list, device, scorer, batch_size: int = 64):
    """
    • Tensor  → 0/1 long tensor 그대로 반환
    • list[str] → CheXbert 로 64개씩 배치 추론 후 0/1 long tensor 반환
    """
    if isinstance(tensor_or_list, torch.Tensor):
        return (tensor_or_list.to(device) == 1).long()

    # ----- list[str] -----
    texts = [(t or "") if isinstance(t, str) else ""        # NaN/None 방어
             for t in tensor_or_list]
    out_chunks = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        preds = scorer.model(chunk)        # [B,14] int64
        out_chunks.append((preds == 1).long())  # 0/1 long
        torch.cuda.empty_cache()                # 메모리 잔여 정리(옵션)

    return torch.cat(out_chunks, dim=0).to(device)


def chex_metrics(gt_bin: torch.Tensor, pr_bin: torch.Tensor) -> dict[str, float]:
    """Return micro/macro/sample precision, recall, F1 metrics."""
    y_true = gt_bin.cpu().numpy()
    y_pred = pr_bin.cpu().numpy()

    p_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    r_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    p_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    p_sample = precision_score(y_true, y_pred, average="samples", zero_division=0)
    r_sample = recall_score(y_true, y_pred, average="samples", zero_division=0)
    f_sample = f1_score(y_true, y_pred, average="samples", zero_division=0)

    return {
        "P_micro": float(p_micro),
        "R_micro": float(r_micro),
        "F1_micro": float(f_micro),
        "P_macro": float(p_macro),
        "R_macro": float(r_macro),
        "F1_macro": float(f_macro),
        "P_sample": float(p_sample),
        "R_sample": float(r_sample),
        "F1_sample": float(f_sample),
    }

def evalcap_metrics(refs: list[str], gens: list[str]) -> dict[str, float]:
    assert len(refs) == len(gens)
    gts = {i: [refs[i]] for i in range(len(refs))}
    res = {i: [gens[i]] for i in range(len(gens))}
    bleu_scores, _  = Bleu(4).compute_score(gts, res)
    meteor_score, _ = Meteor().compute_score(gts, res)
    rouge_score, _  = Rouge().compute_score(gts, res)
    cider_score, _  = Cider().compute_score(gts, res)
    return {
        "BLEU_1" : bleu_scores[0],
        "BLEU_4" : bleu_scores[3],
        "METEOR" : meteor_score,
        "ROUGE_L": rouge_score,
        "CIDEr"  : cider_score,
    }

def compute_batch_nlg_metrics(ref_texts, list_of_preds, num_proc: int = 4) -> dict:
    """
    ref_texts, list_of_preds: 길이가 같은 리스트
    return: {"BLEU_1", "BLEU_4", "METEOR", "ROUGE_L", "CIDEr"}
    """
    return evalcap_metrics(ref_texts, list_of_preds)

def load_actor_critic_from_ckpt(path: str,
                                actor: nn.Module,
                                critic: nn.Module | None = None,
                                device: str = "cuda",
                                strict: bool = True):
    """
    path (.pth) 안에
      - 'actor_state' 키가 있으면 actor 에 로드
      - 'critic_state' 키가 있으면 critic 에 로드
    둘 다 없으면 actor 전체 state_dict 로 가정.
    """
    if not (path and os.path.isfile(path)):
        logging.warning(f"[CKPT] File not found: {path} → 로드 건너뜀")
        return
    ckpt = torch.load(path, map_location=device)
    logging.info(f"[CKPT] Loading weights from {path}")

    # helper – DDP `_orig_mod.` 프리픽스 제거
    def _sanitize(st):
        out = {}
        for k, v in st.items():
            out[k[10:] if k.startswith("_orig_mod.") else k] = v
        return out

    def _resize_for_checkpoint(model, ckpt_state: dict[str, torch.Tensor], weight_keys: list[str]):
        """Resize model token embeddings to match checkpoint vocabulary when needed."""
        vocab_tensor = next((ckpt_state[k] for k in weight_keys if k in ckpt_state), None)
        if vocab_tensor is None:
            return None  # Nothing to do
        ckpt_vocab = vocab_tensor.shape[0]
        current_vocab = model.config.vocab_size if hasattr(model, "config") else None
        if current_vocab is None:
            try:
                current_vocab = model.get_input_embeddings().weight.shape[0]
            except AttributeError:
                return None
        if ckpt_vocab == current_vocab:
            return None
        logging.warning(
            f"[CKPT] Vocab size mismatch detected (checkpoint={ckpt_vocab}, model={current_vocab}). "
            "Temporarily resizing embeddings to load weights."
        )
        model.resize_token_embeddings(ckpt_vocab)
        return current_vocab

    def _restore_vocab(model, target_vocab: int | None):
        if target_vocab is None:
            return
        if hasattr(model, "config") and model.config.vocab_size == target_vocab:
            return
        logging.info(f"[CKPT] Restoring model vocabulary size to {target_vocab}.")
        model.resize_token_embeddings(target_vocab)

    if "actor_state" in ckpt:
        actor_state = _sanitize(ckpt["actor_state"])
    else:
        actor_state = _sanitize(ckpt)

    actor_vocab_restore = _resize_for_checkpoint(
        actor, actor_state, ["transformer.wte.weight", "lm_head.weight"]
    )
    try:
        actor.load_state_dict(actor_state, strict=strict)
    finally:
        _restore_vocab(actor, actor_vocab_restore)

    logging.info("[CKPT] → actor_state 로드 완료" if "actor_state" in ckpt else "[CKPT] → 단독 state_dict 를 actor 에 로드")

    if critic is not None:
        if "critic_state" in ckpt:
            critic_state = _sanitize(ckpt["critic_state"])
            critic_vocab_restore = _resize_for_checkpoint(
                critic.gpt2 if hasattr(critic, "gpt2") else critic,
                critic_state,
                ["gpt2.wte.weight", "transformer.wte.weight"]
            )
            try:
                critic.load_state_dict(critic_state, strict=strict)
            finally:
                if hasattr(critic, "gpt2"):
                    _restore_vocab(critic.gpt2, critic_vocab_restore)
                else:
                    _restore_vocab(critic, critic_vocab_restore)
            logging.info("[CKPT] → critic_state 로드 완료")
        else:
            logging.info("[CKPT] critic_state 키 없음 → critic 로드 생략")

# -------------------------------------------------------------
#  JSON과 동일한 이름의 YAML을 쓰는 헬퍼
# -------------------------------------------------------------
def save_yaml(
    yaml_path: str,
    metrics: dict[str, float],
    cfg_namespace,
    elapsed_time: float | None = None,
):
    """yaml_path 는 .yaml 확장자를 포함한 완전 경로"""
    payload = {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "hparams": vars(cfg_namespace),  # argparse → dict
    }
    if elapsed_time is not None:
        payload["wall_time"] = float(elapsed_time)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def compute_all_metrics(
    full_refs: list[str],
    full_gens: list[str],
    find_refs: list[str],
    find_gens: list[str],
    imp_refs: list[str],
    imp_gens: list[str],
    chex_scorer,
) -> dict[str, float]:
    """Return NLG and CheXbert metrics for all text splits."""

    nlg_find = compute_batch_nlg_metrics(find_refs, find_gens, num_proc=4)
    nlg_imp  = compute_batch_nlg_metrics(imp_refs,  imp_gens,  num_proc=4)
    nlg_full = compute_batch_nlg_metrics(full_refs, full_gens, num_proc=4)

    dev = chex_scorer.model.device
    met_find = chex_metrics(_chex_bin(find_refs, dev, chex_scorer),
                           _chex_bin(find_gens, dev, chex_scorer))
    met_imp  = chex_metrics(_chex_bin(imp_refs, dev, chex_scorer),
                           _chex_bin(imp_gens, dev, chex_scorer))
    met_full = chex_metrics(_chex_bin(full_refs, dev, chex_scorer),
                           _chex_bin(full_gens, dev, chex_scorer))

    metrics = {}
    for prefix, nlg in [
        ("Find", nlg_find),
        ("Imp",  nlg_imp),
        ("Full", nlg_full),
    ]:
        for k, v in nlg.items():
            metrics[f"{prefix}_{k}"] = v

    for prefix, met in [
        ("Find", met_find),
        ("Imp",  met_imp),
        ("Full", met_full),
    ]:
        for k, v in met.items():
            metrics[f"{prefix}_{k}"] = v

    return metrics


def save_results(
    prefix: str,
    rows: list[dict],
    metrics: dict[str, float],
    cfg,
    elapsed_time: float | None = None,
):
    os.makedirs(cfg.log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    json_path = os.path.join(cfg.log_dir, f"{prefix}_{ts}.json")
    with open(json_path, "w") as fp:
        json.dump(rows, fp, indent=2)
    yaml_path = json_path.replace(".json", ".yaml")
    save_yaml(yaml_path, metrics, cfg, elapsed_time)
    logging.info(f"[Eval] saved ⇒ {json_path}")


# ---------------------------------------------------------------------
#           데이터셋 / 전처리
# ---------------------------------------------------------------------
class MIMICCXRDataset(Dataset):
    def __init__(self, csv_file, transform=None, split_mode="train",
                 filter_findings=True, max_images=3, dataset_option="f_or_i", dataset_name="mimiccxr"):
        df = pd.read_csv(csv_file)
        if split_mode == "train":
            df = df[df["split"] == "train"]
        elif split_mode == "validate":
            df = df[df["split"] == "validate"]
        elif split_mode == "test":
            df = df[df["split"] == "test"]
        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")
        df["impression"] = df["impression"].fillna("")
        df["findings"]  = df["findings"].fillna("")
        df["chexbert_labels"] = df["chexbert_labels"].fillna("0;0;0;0;0;0;0;0;0;0;0;0;0;1")
        self.dataset_name = dataset_name.lower()

        if filter_findings:
            if dataset_option == "findings_only":
                df = df[(df["has_findings"]==1)].reset_index(drop=True)
            elif dataset_option == "findings_impression":
                df = df[(df["has_impression"]==1)&(df["has_findings"]==1)].reset_index(drop=True)
            else:
                raise ValueError("dataset_option must be 'findings_only' or 'findings_impression'")
            
        
        keep_rows=[]
        for _,row in df.iterrows():
            if len(row["image_paths"].split(";"))<=max_images:
                keep_rows.append(row)
        self.data = pd.DataFrame(keep_rows).reset_index(drop=True)
        self.transform = transform
        logging.info(f"[Dataset] {split_mode} loaded: {len(self.data)} samples")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        imgs=[]
        for p in row["image_paths"].split(";"):
            if self.dataset_name == "iuxray":
                p = os.path.join("/data/iu_r2gen/iu_xray_r2gen/images/", p)
            else:
                None
            im=Image.open(p).convert("RGB")
            imgs.append(self.transform(im) if self.transform else im)
        return {
            "subject_id":      row["subject_id"],
            "study_id":        row["study_id"],
            "split":           row["split"],
            "images":          imgs,
            "findings":        row["findings"],
            "impression":      row["impression"],
            "report_all":      row["report_all"],
            "chexbert_labels": [int(x) for x in row["chexbert_labels"].split(";")],
        }

def my_collate_fn(batch):
    out = defaultdict(list)
    for samp in batch:
        for k, v in samp.items():
            out[k].append(v)
    return out

# ---------------------------------------------------------------------
#              모델 정의 (Actor / Critic / CheXbert)
# ---------------------------------------------------------------------
CONDITIONS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding"
]
CLASS_MAPPING={0:"Blank",1:"Positive",2:"Negative",3:"Uncertain"}

class SwinImageEncoder(nn.Module):
    def __init__(self, model_name="swinv2_tiny_window16_256.ms_in1k", out_dim=768):
        super().__init__()
        self.swin = timm.create_model(model_name, pretrained=True,
                                      img_size=512, features_only=True)
        self.proj = nn.Conv2d(768, out_dim, 1)
    def forward(self,x):
        f=self.swin(x)[-1].permute(0,3,1,2)
        f=self.proj(f).permute(0,2,3,1).flatten(1,2)
        return f

def pad_sequence(seqs,batch_first=False,padding_value=0.):
    from torch.nn.utils.rnn import pad_sequence as _ps
    return _ps(seqs,batch_first=batch_first,padding_value=padding_value)

def build_multi_image_encoder_states(batch_imgs, encoder, device='cuda', if_frozen=True):
    all_imgs=[im for row in batch_imgs for im in row]
    cnts=[len(r) for r in batch_imgs]
    if len(all_imgs)==0:
        z=torch.zeros(len(batch_imgs),1,encoder.proj.out_channels,device=device)
        m=torch.zeros(len(batch_imgs),1,dtype=torch.long,device=device)
        return z,m
    big=torch.stack(all_imgs).to(device)
    with torch.no_grad() if if_frozen else nullcontext():
        feats=encoder(big)
    split=[feats[start:start+c].reshape(-1,feats.size(-1))
           for start,c in zip(np.cumsum([0]+cnts[:-1]),cnts)]
    pad=pad_sequence(split,True)
    mask=pad_sequence([torch.ones(s.size(0),device=device,dtype=torch.long)
                       for s in split],True)
    return pad,mask


def pad_partials(partials: list, pad_id: int):
    """Pad a list of token tensors to a uniform length."""
    lengths = [p.size(1) for p in partials]
    max_len = max(lengths) if lengths else 1
    B = len(partials)
    device_ = partials[0].device if B > 0 else "cpu"
    out = torch.full((B, max_len), pad_id, dtype=torch.long, device=device_)
    mask = torch.zeros(B, max_len, dtype=torch.long, device=device_)
    for i, seq in enumerate(partials):
        l = seq.size(1)
        out[i, :l] = seq[0]
        mask[i, :l] = 1
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

class MultiImageActor(GPT2LMHeadModel):
    def __init__(self,vocab,hidden_dim=768,
                 swin="swinv2_tiny_window16_256.ms_in1k", gpt2="gpt2"):
        cfg = GPT2Config.from_pretrained(gpt2)
        cfg.add_cross_attention = True
        cfg.vocab_size = vocab
        cfg.n_layer = 30
        super().__init__(cfg)
        self.encoder = SwinImageEncoder(swin, hidden_dim)
        self.aggregator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, cfg.n_head, batch_first=True),
            num_layers=1,
        )
        self.cls_head = nn.Linear(hidden_dim, 14)
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

class MultiImageCritic(nn.Module):
    def __init__(self, hidden_dim=768, swin="swinv2_tiny_window16_256.ms_in1k",
                 gpt2="gpt2"):
        super().__init__()
        self.encoder = SwinImageEncoder(swin, hidden_dim)
        cfg = GPT2Config.from_pretrained(gpt2)
        cfg.add_cross_attention = True
        cfg.n_layer = 6
        self.gpt2 = GPT2LMHeadModel(cfg).transformer  # backbone only
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch_images, partial_tokens, device="cuda",
                attention_mask=None, if_frozen=False, do_sample=False,
                feats_padded=None, mask_padded=None, **kw):
        if feats_padded is None or mask_padded is None:
            feats_padded, mask_padded = build_multi_image_encoder_states(
                batch_images, self.encoder, device, if_frozen=if_frozen
            )
        out=self.gpt2(input_ids=partial_tokens.to(device),
                      attention_mask=attention_mask,
                      encoder_hidden_states=feats_padded,
                      encoder_attention_mask=mask_padded,
                      use_cache=False,return_dict=True)
        return self.value_head(out.last_hidden_state).squeeze(-1)

# ---------- CheXbert ---------- (원본과 동일: 토큰화/RNN + 14-label F1) ----------
class CheXbert(nn.Module):
    def __init__(self, ckpt, device, p=0.1):
        super().__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        cfg = BertConfig.from_pretrained("bert-base-uncased")
        self.bert = BertModel(cfg)
        self.dropout = nn.Dropout(p)
        hs = self.bert.pooler.dense.in_features
        self.linear_heads = nn.ModuleList([nn.Linear(hs, 4) for _ in range(13)])
        self.linear_heads.append(nn.Linear(hs,2))
        st_new = OrderedDict({
            "bert.embeddings.position_ids": torch.arange(
                cfg.max_position_embeddings
            ).expand((1, -1))
        })
        ck = torch.load(ckpt, map_location=device)["model_state_dict"]
        for k, v in ck.items():
            if "bert" in k:
                nk = k.replace("module.bert.", "bert.")
            elif "linear_heads" in k:
                nk = k.replace("module.linear_heads.", "linear_heads.")
            else:
                continue
            st_new[nk] = v
        self.load_state_dict(st_new, strict=False)
        self.to(device).eval()
    @torch.no_grad()
    def forward(self,txts:List[str]):
        txts=[(t or "").strip().replace("\n"," ") for t in txts]
        tok=self.tokenizer(txts,padding='longest',truncation=True,max_length=512,
                           return_tensors='pt').to(self.device)
        cls=self.bert(**tok)[0][:,0,:]
        cls=self.dropout(cls)
        preds=[h(cls).argmax(1) for h in self.linear_heads]
        return torch.stack(preds,1)

class CheXbertScorer:
    def __init__(self,ckpt_path,device='cuda'):
        self.model = CheXbert(ckpt_path, device)

    def _bin(self,p):
        return (p == 1).float()
    @torch.no_grad()
    def score_batch_f1(self,ref,gen,gt,batch=64):
        f = []
        for i in range(0, len(gen), batch):
            gb, rb = gen[i:i + batch], ref[i:i + batch]
            pr = self._bin(self.model(gb))
            gr = self._bin(self.model(rb))
            tp = (pr * gr).sum(1)
            fp = (pr * (1 - gr)).sum(1)
            fn = ((1 - pr) * gr).sum(1)
            f.append(tp / (tp + 0.5 * (fp + fn) + 1e-8))
        return torch.cat(f, 0)
    @torch.no_grad()
    def score_batch_f1_with_gt(self,gen,gt,batch=64):
        f = []
        for i in range(0, len(gen), batch):
            pb = self._bin(self.model(gen[i:i + batch]))
            gb = torch.tensor(gt[i:i + batch], device=self.model.device)
            gb = (gb == 1).float()
            tp = (pb * gb).sum(1)
            fp = (pb * (1 - gb)).sum(1)
            fn = ((1 - pb) * gb).sum(1)
            f.append(tp / (tp + 0.5 * (fp + fn) + 1e-8))
        return torch.cat(f, 0)

# ---------------------------------------------------------------------
#                       Helper Functions
# ---------------------------------------------------------------------
def compute_agg_logits(actor: nn.Module, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return classification logits from aggregator features."""
    agg = actor.aggregator(feats, src_key_padding_mask=(mask == 0))
    mask_f = mask.float().unsqueeze(-1)
    pooled = (agg * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1e-8)
    return actor.cls_head(pooled)


def labels_to_bin(labels) -> torch.Tensor:
    """Convert CheXbert labels (0..3) to binary tensor of shape [N,14]."""
    t = torch.as_tensor(labels, dtype=torch.long)
    return (t == 1).long()


def parse_findings_impression(text: str) -> tuple[str, str]:
    """Split generated text into findings and impression parts."""
    text = text.replace("  ", " ").replace("  ", " ").replace("  ", " ")
    match_f = re.search(r"Findings:\s*(.*?)(?:Impression:|$)", text, flags=re.IGNORECASE)
    findings = match_f.group(1).strip() if match_f else text.strip()
    match_i = re.search(r"Impression:\s*(.*)", text, flags=re.IGNORECASE)
    impression = match_i.group(1).strip() if match_i else ""
    # parts = re.split(r"[Ii]mpression\s*:", text, maxsplit=1)
    # findings = parts[0].strip()
    # impression = parts[1].strip() if len(parts) == 2 else ""
    return findings, impression

# ---------------------------------------------------------------------
#              평가 함수 (one_shot / energy_based / cfgl / vfgs)
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(
        actor,
        data_loader,
        tokenizer,
        device,
        chex_scorer,
        cfg,
        max_seq_len=32,
        test_subset_frac=0.2,
        num_beams=1
):
    start_time = time.perf_counter()
    ddp = isinstance(actor, DDP)
    base_actor = actor.module if ddp else actor

    hf_model = GPT2BeamSearchWrapper(base_actor)
    hf_model.eval()
    hf_model.to(device)

    pairs_local = []
    all_refs = []
    all_gens = []
    all_chex_gt = []
    all_partial_prompts = []

    all_class_preds = []
    all_class_gts = []
    results_per_rank = []

    all_find_refs, all_find_gens, all_imp_refs, all_imp_gens = [], [], [], []

    total_test = len(data_loader)
    max_test_batches = int(test_subset_frac * total_test)
    ddp_rank0 = (not ddp) or (ddp and int(os.environ.get('RANK', '0')) == 0)
    if ddp_rank0:
        iterator = tqdm(enumerate(data_loader),
                        total=min(max_test_batches, total_test),
                        desc="Evaluating (Beam)")
    else:
        iterator = enumerate(data_loader)

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
            ref_str = f"Findings: {findings_} Impression: {impress_}"
            ref_texts.append(ref_str)
        all_refs.extend(ref_texts)
        all_find_refs.extend(batch_data["findings"])
        all_imp_refs.extend(batch_data["impression"])

        all_chex_gt.extend(batch_data["chexbert_labels"])

        feats_padded, mask_padded = build_multi_image_encoder_states(
            batch_data["images"], base_actor.encoder, device=device, if_frozen=True,
        )
        cls_logits = compute_agg_logits(base_actor, feats_padded, mask_padded)
        preds_bin = (torch.sigmoid(cls_logits) >= 0.5).long()
        all_class_preds.append(preds_bin.cpu())

        gt_bin = labels_to_bin(batch_data["chexbert_labels"])
        all_class_gts.append(gt_bin)

        partial_prompts = []
        for i in range(B):
            row = preds_bin[i]
            parts = []
            for idx_cond, cond_name in enumerate(CONDITIONS):
                # if row[idx_cond]==1 => Positive else Negative
                class_str = "Positive" if row[idx_cond].item() == 1 else "Blank"
                cond_token = cond_name.replace(" ", "_")
                parts.append(f"<{cond_token}>:<{class_str}>")
            classification_line = ";".join(parts) + " <cls_end>"
            partial_prompts.append(classification_line)
        all_partial_prompts.extend(partial_prompts)


        prefix_text = "Findings:"
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        input_ids = (
            torch.tensor(prefix_ids, dtype=torch.long, device=device)
            .unsqueeze(0)
            .repeat(B, 1)
        )
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

        gen_full_texts = []
        for i in range(generated.size(0)):
            new_tokens = generated[i, :]
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            new_text = new_text.replace("  ", " ").replace("  ", " ").replace("  ", " ")
            f_txt, i_txt = parse_findings_impression(new_text)
            # full_txt = f"Findings: {f_txt} Impression: {i_txt}"
            gen_full_texts.append(new_text)

            results_per_rank.append({
                "subject_id": str(batch_data["subject_id"][i]),
                "study_id"  : str(batch_data["study_id"][i]),
                "ref"       : ref_texts[i],
                "gen"       : new_text,
                "split"     : str(batch_data["split"][i])
            })
            all_find_gens.append(f_txt)
            all_imp_gens.append(i_txt)

        all_gens.extend(gen_full_texts)
        pairs_local.extend(list(zip(ref_texts, gen_full_texts)))

    if len(all_class_preds) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    all_class_preds_tensor = torch.cat(all_class_preds, dim=0)
    all_class_gts_tensor = torch.cat(all_class_gts, dim=0)

    # --- (신규)  :   all‑reduce 로 TP/FP/FN 합산 후 F1 계산 ---
    tp_local = ((all_class_preds_tensor == 1) & (all_class_gts_tensor == 1)).sum().to(device)
    fp_local = ((all_class_preds_tensor == 1) & (all_class_gts_tensor == 0)).sum().to(device)
    fn_local = ((all_class_preds_tensor == 0) & (all_class_gts_tensor == 1)).sum().to(device)

    if torch.distributed.is_initialized():
        for t in (tp_local, fp_local, fn_local):
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)

    classification_f1 = tp_local / (tp_local + 0.5 * (fp_local + fn_local) + 1e-8)
    classification_f1 = classification_f1.item()
    # ----------------------------------------------------------

    # ---------- DDP 결과 모으기 ----------
    if torch.distributed.is_initialized():      # ddp인 경우
        world_size = torch.distributed.get_world_size()
        gathered = [None] * world_size
        torch.distributed.barrier()             # 동기화
        torch.distributed.gather_object(
            results_per_rank,
            gathered if ddp_rank0 else None,    # rank0만 받음
            dst=0
        )
        if ddp_rank0:
            merged_results = [item for sub in gathered for item in sub]
    else:                                       # 단일 프로세스
        merged_results = results_per_rank
    
    if ddp_rank0:
        metrics = compute_all_metrics(
            all_refs,
            all_gens,
            all_find_refs,
            all_find_gens,
            all_imp_refs,
            all_imp_gens,
            chex_scorer,
        )
        elapsed = time.perf_counter() - start_time
        save_results("eval_output", merged_results, metrics, cfg, elapsed)

        for i in range(min(2, len(all_gens))):
            logging.info(f"--- Example {i} ---")
            logging.info(f"Prompt: {all_partial_prompts[i]}")
            logging.info(f"Generated (full): {all_gens[i]}")
            logging.info(f"Generated (only impression): {all_imp_gens[i]}")
            logging.info(f"Ref: {all_refs[i]}")

        return (
            classification_f1,
            metrics["Full_BLEU_1"],
            metrics["Full_BLEU_4"],
            metrics["Full_METEOR"],
            metrics["Full_ROUGE_L"],
            metrics["Full_F1_micro"],
        )

    return classification_f1, 0.0, 0.0, 0.0, 0.0, 0.0


@torch.no_grad()
def evaluate_model_flexible_next_debug(
    actor: nn.Module,
    critic: nn.Module,
    data_loader: DataLoader,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    chex_scorer,
    cfg,
    *,
    forced_next_count: int = 0,
    max_seq_len: int = 100,
    test_subset_frac: float = 0.2,
    impr_n: int = 1,
    find_top_p: float = 0.9,
    impr_top_p: float = 0.9,
    findings_temperature: float = 1.0,
    impression_temperature: float = 0.3,
):
    """Same as ``evaluate_model_flexible_next`` (initial debug version)."""

    return evaluate_model_flexible_next(
        actor,
        critic,
        data_loader,
        tokenizer,
        device,
        chex_scorer,
        cfg,
        forced_next_count=forced_next_count,
        max_seq_len=max_seq_len,
        test_subset_frac=test_subset_frac,
        impr_n=impr_n,
        find_top_p=find_top_p,
        impr_top_p=impr_top_p,
        findings_temperature=findings_temperature,
        impression_temperature=impression_temperature,
    )


@torch.no_grad()
def evaluate_model_flexible_next_debug_logging(
    actor: nn.Module,
    critic: nn.Module,
    data_loader: DataLoader,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    chex_scorer,
    cfg,
    logit_actor: nn.Module,
    *,
    forced_next_count: int = 0,
    max_seq_len: int = 100,
    test_subset_frac: float = 0.2,
    impr_n: int = 1,
    find_top_p: float = 0.9,
    impr_top_p: float = 0.9,
    findings_temperature: float = 1.0,
    impression_temperature: float = 0.3,
):
    """Debugging variant of flexible-next decoding with log tracking."""

    start_time = time.perf_counter()
    ddp_gen = isinstance(actor, DDP)
    ddp_c = isinstance(critic, DDP)
    ddp_log = isinstance(logit_actor, DDP)
    ddp = ddp_gen
    gen_actor = actor.module if ddp_gen else actor
    base_critic = critic.module if ddp_c else critic
    log_actor = logit_actor.module if ddp_log else logit_actor
    gen_actor.eval().to(device)
    base_critic.eval().to(device)
    log_actor.eval().to(device)
    gen_wrap = GPT2GenerationWrapper(gen_actor).eval().to(device)

    eos_id = tokenizer.eos_token_id or tokenizer.bos_token_id
    next_id = (
        tokenizer.convert_tokens_to_ids("<next>")
        if "<next>" in tokenizer.get_vocab() else None
    )
    impr_id = (
        tokenizer.convert_tokens_to_ids("<impression>")
        if "<impression>" in tokenizer.get_vocab() else None
    )

    if forced_next_count and next_id is None:
        raise ValueError("forced_next_count > 0 but <next> token not in vocab")

    prefix_ids = torch.tensor(
        tokenizer.encode("Findings:", add_special_tokens=False), device=device
    )


    def _sample(logits, top_p, temp):
        logits = logits / temp
        probs = torch.softmax(logits, -1)
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cdf = sorted_p.cumsum(-1)
        mask = cdf > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_p = sorted_p.masked_fill(mask, 0.0)
        sorted_p = sorted_p / sorted_p.sum(-1, keepdim=True)
        idx = torch.multinomial(sorted_p, 1).squeeze(-1)
        return sorted_i.gather(-1, idx.unsqueeze(-1)).item()

    cls_pred_list, cls_gt_list = [], []
    results_per_rank = []
    all_refs = []
    all_gens = []
    ref_findings_all, ref_impressions_all = [], []
    gen_findings_all, gen_impressions_all = [], []

    max_batches = int(len(data_loader) * test_subset_frac)
    iterator = tqdm(
        enumerate(data_loader),
        total=max_batches,
        desc=f"Eval-Next-Debug{forced_next_count}",
        disable=ddp and int(os.environ.get("RANK", 0)) != 0,
    )

    for b_idx, batch in iterator:
        if b_idx >= max_batches:
            break
        B = len(batch["report_all"])
        if B == 0:
            continue

        ref_findings_all.extend(batch["findings"])
        ref_impressions_all.extend(batch["impression"])

        feats_gen, mask_gen = build_multi_image_encoder_states(
            batch["images"], gen_actor.encoder, device=device, if_frozen=True
        )
        feats_log, mask_log = build_multi_image_encoder_states(
            batch["images"], log_actor.encoder, device=device, if_frozen=True
        )
        cls_logits = compute_agg_logits(gen_actor, feats_gen, mask_gen)
        preds_bin = (torch.sigmoid(cls_logits) >= 0.5).long()
        cls_pred_list.append(preds_bin.cpu())
        cls_gt_list.append(labels_to_bin(batch["chexbert_labels"]))

        parts = [prefix_ids.unsqueeze(0).clone() for _ in range(B)]
        n_next = [0] * B
        done = [False] * B

        logit_next = [[] for _ in range(B)]
        log_prob_next = [[] for _ in range(B)]
        logit_impr = [[] for _ in range(B)]
        log_prob_impr = [[] for _ in range(B)]

        for _ in range(max_seq_len):
            active = [i for i, d in enumerate(done) if not d]
            if not active:
                break

            toks, _ = pad_partials([parts[i] for i in active], tokenizer.pad_token_id)

            gen_logits_step = gen_actor(
                batch_images=None,
                partial_tokens=toks,
                device=device,
                do_sample=False,
                feats_padded=feats_gen[active],
                mask_padded=mask_gen[active],
            )
            last = gen_logits_step[:, -1, :]

            for ai, bi in enumerate(active):

                tok = _sample(last[ai], find_top_p, findings_temperature)
                if n_next[bi] < forced_next_count:
                    if (tok == eos_id or (impr_id and tok == impr_id)) and next_id is not None:
                        tok = next_id
                    if tok == next_id:
                        n_next[bi] += 1
                else:
                    if not (tok == eos_id or (impr_id and tok == impr_id)):
                        tok = impr_id if impr_id is not None else eos_id
                    done[bi] = True

                parts[bi] = torch.cat([parts[bi], toks.new_tensor([[tok]])], 1)
                if tok == eos_id or (impr_id and tok == impr_id):
                    done[bi] = True

        # ----- compute per-sentence <next> logits after generation -----
        for i in range(B):
            seq = parts[i].squeeze(0)
            tokens_after_prefix = seq[len(prefix_ids):]
            start = 0
            for j, t_id in enumerate(tokens_after_prefix.tolist()):
                if t_id in {next_id, impr_id, eos_id}:
                    if j > start:
                        partial = torch.cat([
                            prefix_ids,
                            tokens_after_prefix[start:j],
                        ])
                        logits = log_actor(
                            batch_images=None,
                            partial_tokens=partial.unsqueeze(0),
                            device=device,
                            do_sample=False,
                            feats_padded=feats_log[i:i+1],
                            mask_padded=mask_log[i:i+1],
                        )
                        lvec = logits[0, -1]
                        lp = torch.log_softmax(lvec, -1)
                        if t_id == next_id and next_id is not None:
                            logit_next[i].append(float(lvec[next_id]))
                            log_prob_next[i].append(float(lp[next_id]))
                            logit_impr[i].append(float(lvec[impr_id]))
                            log_prob_impr[i].append(float(lp[impr_id]))
                        elif t_id == impr_id and impr_id is not None:
                            logit_next[i].append(float(lvec[next_id]))
                            log_prob_next[i].append(float(lp[next_id]))
                            logit_impr[i].append(float(lvec[impr_id]))
                            log_prob_impr[i].append(float(lp[impr_id]))
                    if t_id == next_id:
                        start = j + 1
                        continue
                    break

        dec_findings = [
            tokenizer.decode(p.squeeze(0), skip_special_tokens=True)
            .replace("  ", " ")
            .replace("  ", " ")
            .replace("  ", " ")
            for p in parts
        ]
        gen_findings_all.extend(dec_findings)

        for i in range(B):
            prefix_text = dec_findings[i].strip()
            if impr_id is not None and "<impression>" not in prefix_text.lower():
                prefix_text += " <impression>"

            inp = torch.tensor(
                tokenizer.encode(prefix_text, add_special_tokens=False),
                device=device,
            ).unsqueeze(0)
            att = torch.ones_like(inp)

            gen_wrap.set_encoder_states(feats_gen[i:i+1], mask_gen[i:i+1])
            cand_vals = []
            cand_seqs = []
            for _ in range(impr_n):
                out = gen_wrap.generate(
                    input_ids=inp,
                    attention_mask=att,
                    max_new_tokens=max_seq_len,
                    do_sample=True,
                    top_p=impr_top_p,
                    temperature=impression_temperature,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_id,
                    use_cache=False,
                )
                seq = out[0]
                cand_seqs.append(seq)
                c_att = (seq != tokenizer.pad_token_id).long().unsqueeze(0)
                c_pos = c_att.cumsum(1) - 1
                val = base_critic(
                    batch_images=None,
                    partial_tokens=seq.unsqueeze(0),
                    feats_padded=feats_gen[i:i+1],
                    mask_padded=mask_gen[i:i+1],
                    device=device,
                    attention_mask=c_att,
                    position_ids=c_pos,
                )[:, -1]
                cand_vals.append(val.item())
            gen_wrap.clear_encoder_states()

            best_idx = int(torch.tensor(cand_vals).argmax())
            best_seq = cand_seqs[best_idx]

            out_str = tokenizer.decode(best_seq, skip_special_tokens=True).replace("  ", " ").replace("  ", " ").replace("  ", " ")
            m = re.search(r"[Ii]mpression:\s*(.*)", out_str)
            gen_imp = m.group(1).strip() if m else ""
            gen_impressions_all.append(gen_imp)

            full_ref = f"Findings: {batch['findings'][i]} Impression: {batch['impression'][i]}"
            all_refs.append(full_ref)
            all_gens.append(out_str)
            results_per_rank.append({
                "subject_id": str(batch["subject_id"][i]),
                "study_id": str(batch["study_id"][i]),
                "split": str(batch["split"][i]),
                "ref": full_ref,
                "gen": out_str,
                "logit_next": logit_next[i],
                "log_prob_next": log_prob_next[i],
                "logit_impression": logit_impr[i],
                "log_prob_impression": log_prob_impr[i],
            })

    if cls_pred_list:
        cls_pred_cat = torch.cat(cls_pred_list)
        cls_gt_cat = torch.cat(cls_gt_list)
        cls_f1, cls_P, cls_R = compute_pos_only_f1(cls_pred_cat, cls_gt_cat)
    else:
        cls_f1, cls_P, cls_R = 0.0, 0.0, 0.0

    logging.info(
        f"[FlexNextDebug] Class-F1={cls_f1:.3f}  P={cls_P:.3f}  R={cls_R:.3f}"
    )

    rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
    if rank0:
        metrics = compute_all_metrics(
            all_refs,
            all_gens,
            ref_findings_all,
            gen_findings_all,
            ref_impressions_all,
            gen_impressions_all,
            chex_scorer,
        )
        elapsed = time.perf_counter() - start_time
        save_results("eval_flexnext_debug", results_per_rank, metrics, cfg, elapsed)

        for i in range(min(2, len(gen_findings_all))):
            logging.info(f"[FlexNextDebug Sample {i}]  GEN-FIND: {gen_findings_all[i]}")
            logging.info(f"[FlexNextDebug Sample {i}]  REF-FIND: {ref_findings_all[i]}")
        for i in range(min(2, len(gen_impressions_all))):
            logging.info(f"[FlexNextDebug Sample {i}]  GEN-IMPR: {gen_impressions_all[i]}")
            logging.info(f"[FlexNextDebug Sample {i}]  REF-IMPR: {ref_impressions_all[i]}")

        return (
            cls_f1,
            metrics["Full_BLEU_1"],
            metrics["Full_BLEU_4"],
            metrics["Full_METEOR"],
            metrics["Full_ROUGE_L"],
            metrics["Full_F1_micro"],
        )

    return cls_f1, 0.0, 0.0, 0.0, 0.0, 0.0


@torch.no_grad()
def evaluate_model_flexible_next(
    actor: nn.Module,
    critic: nn.Module,
    data_loader: DataLoader,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    chex_scorer,
    cfg,
    *,
    forced_next_count: int = 0,
    max_seq_len: int = 100,
    test_subset_frac: float = 0.2,
    impr_n: int = 1,
    find_top_p: float = 0.9,
    impr_top_p: float = 0.9,
    findings_temperature: float = 1.0,
    impression_temperature: float = 0.5,
):
    """Evaluate with the flexible-next decoding strategy.

    After generating ``forced_next_count`` ``<next>`` tokens, the findings and
    impression sections are decoded with top-p sampling. The impression portion
    (from ``<impression>`` to ``<eos>``) is sampled ``impr_n`` times and scored
    by the critic. The highest valued sequence becomes the final generation.
    """

    start_time = time.perf_counter()
    ddp = isinstance(actor, DDP)
    ddp_c = isinstance(critic, DDP)
    base_actor = actor.module if ddp else actor
    base_critic = critic.module if ddp_c else critic
    base_actor.eval().to(device)
    base_critic.eval().to(device)
    gen_wrap = GPT2GenerationWrapper(base_actor).eval().to(device)

    eos_id = tokenizer.eos_token_id or tokenizer.bos_token_id
    next_id = (
        tokenizer.convert_tokens_to_ids("<next>")
        if "<next>" in tokenizer.get_vocab() else None
    )
    impr_id = (
        tokenizer.convert_tokens_to_ids("<impression>")
        if "<impression>" in tokenizer.get_vocab() else None
    )

    if forced_next_count and next_id is None:
        raise ValueError("forced_next_count > 0 but <next> token not in vocab")

    prefix_ids = torch.tensor(
        tokenizer.encode("Findings:", add_special_tokens=False), device=device
    )

    def _sample(logits, top_p, temp):
        logits = logits / temp
        probs = torch.softmax(logits, -1)
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cdf = sorted_p.cumsum(-1)
        mask = cdf > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_p = sorted_p.masked_fill(mask, 0.0)
        sorted_p = sorted_p / sorted_p.sum(-1, keepdim=True)
        idx = torch.multinomial(sorted_p, 1).squeeze(-1)
        return sorted_i.gather(-1, idx.unsqueeze(-1)).item()

    cls_pred_list, cls_gt_list = [], []
    results_per_rank = []
    all_refs = []
    all_gens = []
    ref_findings_all, ref_impressions_all = [], []
    gen_findings_all, gen_impressions_all = [], []

    max_batches = int(len(data_loader) * test_subset_frac)
    iterator = tqdm(
        enumerate(data_loader),
        total=max_batches,
        desc=f"Eval-Next-Forcing{forced_next_count}",
        disable=ddp and int(os.environ.get("RANK", 0)) != 0,
    )

    for b_idx, batch in iterator:
        if b_idx >= max_batches:
            break
        B = len(batch["report_all"])
        if B == 0:
            continue

        ref_findings_all.extend(batch["findings"])
        ref_impressions_all.extend(batch["impression"])

        feats, mask = build_multi_image_encoder_states(
            batch["images"], base_actor.encoder, device=device, if_frozen=True
        )
        cls_logits = compute_agg_logits(base_actor, feats, mask)
        preds_bin = (torch.sigmoid(cls_logits) >= 0.5).long()
        cls_pred_list.append(preds_bin.cpu())
        cls_gt_list.append(labels_to_bin(batch["chexbert_labels"]))

        parts = [prefix_ids.unsqueeze(0).clone() for _ in range(B)]
        n_next = [0] * B
        done = [False] * B

        for _ in range(max_seq_len):
            active = [i for i, d in enumerate(done) if not d]
            if not active:
                break

            toks, _ = pad_partials([parts[i] for i in active], tokenizer.pad_token_id)
            imgs = [batch["images"][i] for i in active]
            logits = base_actor(imgs, toks, device=device, do_sample=False)
            last = logits[:, -1, :]
            # nxt = logits[:, -1].argmax(-1)

            for ai, bi in enumerate(active):
                tok = _sample(last[ai], find_top_p, findings_temperature)
                # tok = nxt[ai].item()

                if n_next[bi] < forced_next_count:
                    if (tok == eos_id or (impr_id and tok == impr_id)) and next_id is not None:
                        tok = next_id
                    if tok == next_id:
                        n_next[bi] += 1
                else:
                    if not (tok == eos_id or (impr_id and tok == impr_id)):
                        tok = impr_id if impr_id is not None else eos_id
                    done[bi] = True

                parts[bi] = torch.cat([parts[bi], toks.new_tensor([[tok]])], 1)
                if tok == eos_id or (impr_id and tok == impr_id):
                    done[bi] = True

        dec_findings = [tokenizer.decode(p.squeeze(0), skip_special_tokens=True).replace("  ", " ").replace("  ", " ").replace("  ", " ") for p in parts]
        gen_findings_all.extend(dec_findings)

        for i in range(B):
            prefix_text = dec_findings[i].strip()
            if impr_id is not None and "<impression>" not in prefix_text.lower():
                prefix_text += " <impression>"

            inp = torch.tensor(
                tokenizer.encode(prefix_text, add_special_tokens=False),
                device=device,
            ).unsqueeze(0)
            att = torch.ones_like(inp)

            gen_wrap.set_encoder_states(feats[i:i+1], mask[i:i+1])
            cand_vals = []
            cand_seqs = []
            for _ in range(impr_n):
                out = gen_wrap.generate(
                    input_ids=inp,
                    attention_mask=att,
                    max_new_tokens=max_seq_len,
                    do_sample=True,
                    top_p=impr_top_p,
                    temperature=impression_temperature,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_id,
                    use_cache=False,
                )
                seq = out[0]
                cand_seqs.append(seq)
                c_att = (seq != tokenizer.pad_token_id).long().unsqueeze(0)
                c_pos = c_att.cumsum(1) - 1
                val = base_critic(
                    batch_images=None,
                    partial_tokens=seq.unsqueeze(0),
                    feats_padded=feats[i:i+1],
                    mask_padded=mask[i:i+1],
                    device=device,
                    attention_mask=c_att,
                    position_ids=c_pos,
                )[:, -1]
                cand_vals.append(val.item())
            gen_wrap.clear_encoder_states()

            best_idx = int(torch.tensor(cand_vals).argmax())
            best_seq = cand_seqs[best_idx]

            out_str = tokenizer.decode(best_seq, skip_special_tokens=True).replace("  ", " ").replace("  ", " ").replace("  ", " ")
            m = re.search(r"[Ii]mpression:\s*(.*)", out_str)
            gen_imp = m.group(1).strip() if m else ""
            gen_impressions_all.append(gen_imp)

            full_ref = f"Findings: {batch['findings'][i]} Impression: {batch['impression'][i]}"
            all_refs.append(full_ref)
            all_gens.append(out_str)
            results_per_rank.append({
                "subject_id": str(batch["subject_id"][i]),
                "study_id": str(batch["study_id"][i]),
                "split": str(batch["split"][i]),
                "ref": full_ref,
                "gen": out_str,
            })

    if cls_pred_list:
        cls_pred_cat = torch.cat(cls_pred_list)
        cls_gt_cat = torch.cat(cls_gt_list)
        cls_f1, cls_P, cls_R = compute_pos_only_f1(cls_pred_cat, cls_gt_cat)
    else:
        cls_f1, cls_P, cls_R = 0.0, 0.0, 0.0

    logging.info(
        f"[FlexNext] Class-F1={cls_f1:.3f}  P={cls_P:.3f}  R={cls_R:.3f}"
    )

    rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
    if rank0:
        metrics = compute_all_metrics(
            all_refs,
            all_gens,
            ref_findings_all,
            gen_findings_all,
            ref_impressions_all,
            gen_impressions_all,
            chex_scorer,
        )
        elapsed = time.perf_counter() - start_time
        save_results("eval_flexnext", results_per_rank, metrics, cfg, elapsed)

        for i in range(min(2, len(gen_findings_all))):
            logging.info(f"[FlexNext Sample {i}]  GEN-FIND: {gen_findings_all[i]}")
            logging.info(f"[FlexNext Sample {i}]  REF-FIND: {ref_findings_all[i]}")
        for i in range(min(2, len(gen_impressions_all))):
            logging.info(f"[FlexNext Sample {i}]  GEN-IMPR: {gen_impressions_all[i]}")
            logging.info(f"[FlexNext Sample {i}]  REF-IMPR: {ref_impressions_all[i]}")

        return (
            cls_f1,
            metrics["Full_BLEU_1"],
            metrics["Full_BLEU_4"],
            metrics["Full_METEOR"],
            metrics["Full_ROUGE_L"],
            metrics["Full_F1_micro"],
        )

    return cls_f1, 0.0, 0.0, 0.0, 0.0, 0.0

@torch.no_grad()
def flex_next_whole_bon(
    actor: nn.Module,
    critic: nn.Module,
    data_loader: DataLoader,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    chex_scorer,
    cfg,
    *,
    forced_next_count: int = 0,
    max_seq_len: int = 100,
    test_subset_frac: float = 0.2,
    n_samples: int = 8,
    find_top_p: float = 0.9,
    impr_top_p: float = 0.9,
    find_temp: float = 1.0,
    impr_temp: float = 0.5,
):
    """Best-of-N decoding with flexible-next over the entire sequence."""

    start_time = time.perf_counter()
    ddp = isinstance(actor, DDP)
    ddp_c = isinstance(critic, DDP)
    base_actor = actor.module if ddp else actor
    base_critic = critic.module if ddp_c else critic
    base_actor.eval().to(device)
    base_critic.eval().to(device)
    gen_wrap = GPT2GenerationWrapper(base_actor).eval().to(device)

    eos_id = tokenizer.eos_token_id or tokenizer.bos_token_id
    next_id = (
        tokenizer.convert_tokens_to_ids("<next>")
        if "<next>" in tokenizer.get_vocab() else None
    )
    impr_id = (
        tokenizer.convert_tokens_to_ids("<impression>")
        if "<impression>" in tokenizer.get_vocab() else None
    )

    if forced_next_count and next_id is None:
        raise ValueError("forced_next_count > 0 but <next> token not in vocab")

    prefix_ids = torch.tensor(
        tokenizer.encode("Findings:", add_special_tokens=False), device=device
    )

    def _sample(logits, top_p, temp):
        logits = logits / temp
        probs = torch.softmax(logits, -1)
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cdf = sorted_p.cumsum(-1)
        mask = cdf > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_p = sorted_p.masked_fill(mask, 0.0)
        sorted_p = sorted_p / sorted_p.sum(-1, keepdim=True)
        idx = torch.multinomial(sorted_p, 1).squeeze(-1)
        return sorted_i.gather(-1, idx.unsqueeze(-1)).item()

    results_per_rank = []
    all_refs = []
    all_gens = []
    ref_findings_all, ref_impressions_all = [], []
    gen_findings_all, gen_impressions_all = [], []
    cls_pred_list, cls_gt_list = [], []

    max_batches = int(len(data_loader) * test_subset_frac)
    iterator = tqdm(
        enumerate(data_loader),
        total=max_batches,
        desc="FlexNextBON",
        disable=ddp and int(os.environ.get("RANK", 0)) != 0,
    )

    for b_idx, batch in iterator:
        if b_idx >= max_batches:
            break
        B = len(batch["report_all"])
        if B == 0:
            continue

        feats, mask = build_multi_image_encoder_states(
            batch["images"], base_actor.encoder, device=device, if_frozen=True
        )
        cls_logits = compute_agg_logits(base_actor, feats, mask)
        preds_bin = (torch.sigmoid(cls_logits) >= 0.5).long()
        cls_pred_list.append(preds_bin.cpu())
        cls_gt_list.append(labels_to_bin(batch["chexbert_labels"]))

        ref_findings_all.extend(batch["findings"])
        ref_impressions_all.extend(batch["impression"])

        cand_vals = torch.zeros(B, n_samples, device=device)
        cand_seqs = [[None] * n_samples for _ in range(B)]

        for n in range(n_samples):
            parts = [prefix_ids.unsqueeze(0).clone() for _ in range(B)]
            n_next = [0] * B
            done = [False] * B
            after_impr = [False] * B

            for _ in range(max_seq_len):
                active = [i for i, d in enumerate(done) if not d]
                if not active:
                    break

                toks, _ = pad_partials([parts[i] for i in active], tokenizer.pad_token_id)
                logits = base_actor(
                    batch_images=None,
                    partial_tokens=toks,
                    feats_padded=feats[active],
                    mask_padded=mask[active],
                    device=device,
                    do_sample=False,
                )
                last = logits[:, -1, :]

                for ai, bi in enumerate(active):
                    tp = impr_top_p if after_impr[bi] else find_top_p
                    tm = impr_temp if after_impr[bi] else find_temp
                    tok = _sample(last[ai], tp, tm)

                    if not after_impr[bi]:
                        if n_next[bi] < forced_next_count:
                            if (tok == eos_id or (impr_id and tok == impr_id)) and next_id is not None:
                                tok = next_id
                            if tok == next_id:
                                n_next[bi] += 1
                        else:
                            if not (tok == eos_id or (impr_id and tok == impr_id)):
                                tok = impr_id if impr_id is not None else eos_id
                            done[bi] = True
                        if tok == impr_id:
                            after_impr[bi] = True
                        if tok == eos_id:
                            done[bi] = True
                    else:
                        if tok == eos_id:
                            done[bi] = True

                    parts[bi] = torch.cat([parts[bi], toks.new_tensor([[tok]])], 1)

            seqs_full = []
            vals = []
            for i in range(B):
                inp = parts[i].squeeze(0)
                att = (inp != tokenizer.pad_token_id).long().unsqueeze(0)
                gen_wrap.set_encoder_states(feats[i:i+1], mask[i:i+1])
                out = gen_wrap.generate(
                    input_ids=inp.unsqueeze(0),
                    attention_mask=att,
                    max_new_tokens=max_seq_len,
                    do_sample=True,
                    top_p=impr_top_p,
                    temperature=impr_temp,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_id,
                    use_cache=False,
                )
                gen_wrap.clear_encoder_states()
                seq = out[0]
                seqs_full.append(seq)

                a = (seq != tokenizer.pad_token_id).long().unsqueeze(0)
                pos = a.cumsum(1) - 1
                val = base_critic(
                    batch_images=None,
                    partial_tokens=seq.unsqueeze(0),
                    feats_padded=feats[i:i+1],
                    mask_padded=mask[i:i+1],
                    device=device,
                    attention_mask=a,
                    position_ids=pos,
                )[:, -1]
                vals.append(val.item())

            for i in range(B):
                cand_seqs[i][n] = seqs_full[i]
                cand_vals[i, n] = vals[i]

        best_idx = cand_vals.argmax(dim=1)
        for i in range(B):
            seq = cand_seqs[i][best_idx[i].item()]
            text = tokenizer.decode(seq, skip_special_tokens=True)
            text = text.replace("  ", " ").replace("  ", " ").replace("  ", " ")
            f_txt, i_txt = parse_findings_impression(text)

            gen_findings_all.append(f_txt)
            gen_impressions_all.append(i_txt)

            full_ref = f"Findings: {batch['findings'][i]} Impression: {batch['impression'][i]}"
            all_refs.append(full_ref)
            all_gens.append(text)
            results_per_rank.append({
                "subject_id": str(batch["subject_id"][i]),
                "study_id": str(batch["study_id"][i]),
                "split": str(batch["split"][i]),
                "ref": full_ref,
                "gen": text,
            })

    if cls_pred_list:
        cls_pred_cat = torch.cat(cls_pred_list)
        cls_gt_cat = torch.cat(cls_gt_list)
        cls_f1, cls_P, cls_R = compute_pos_only_f1(cls_pred_cat, cls_gt_cat)
    else:
        cls_f1, cls_P, cls_R = 0.0, 0.0, 0.0

    rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
    if rank0:
        metrics = compute_all_metrics(
            all_refs,
            all_gens,
            ref_findings_all,
            gen_findings_all,
            ref_impressions_all,
            gen_impressions_all,
            chex_scorer,
        )
        elapsed = time.perf_counter() - start_time
        save_results("eval_flexnext_bon", results_per_rank, metrics, cfg, elapsed)

        for i in range(min(2, len(gen_findings_all))):
            logging.info(f"[FlexNextBON Sample {i}] GEN-FIND: {gen_findings_all[i]}")
            logging.info(f"[FlexNextBON Sample {i}] REF-FIND: {ref_findings_all[i]}")
        for i in range(min(2, len(gen_impressions_all))):
            logging.info(f"[FlexNextBON Sample {i}] GEN-IMPR: {gen_impressions_all[i]}")
            logging.info(f"[FlexNextBON Sample {i}] REF-IMPR: {ref_impressions_all[i]}")

        return (
            cls_f1,
            metrics["Full_BLEU_1"],
            metrics["Full_BLEU_4"],
            metrics["Full_METEOR"],
            metrics["Full_ROUGE_L"],
            metrics["Full_F1_micro"],
        )

    return cls_f1, 0.0, 0.0, 0.0, 0.0, 0.0


 
class ImpressionTemperatureProcessor(LogitsProcessor):
    """
    During generation keeps `temp_find` until the model has emitted the token
    sequence "<impression>" (case-insensitive) and then switches to
    `temp_impr` for every subsequent token.
    """
    def __init__(self, tokenizer, temp_find: float = 1.0, temp_impr: float = 0.3):
        self.tok = tokenizer
        self.t_find = temp_find
        self.t_impr = temp_impr
        # token IDs for the string "<impression>"
        self.marker_ids = tokenizer.encode("<impression>", add_special_tokens=False)
 
    def _seen_marker(self, input_ids):
        """Check if last len(marker_ids) tokens match <impression>."""
        m = len(self.marker_ids)
        if input_ids.size(1) < m:
            return False
        # Compare the last m tokens
        tail = input_ids[0, -m:]
        marker = torch.tensor(self.marker_ids, device=input_ids.device)
        return torch.equal(tail, marker)
 
    def __call__(self, input_ids, scores):
        # scores shape: [batch, vocab]
        if self._seen_marker(input_ids):
            # After <impression> is fully emitted
            scores = scores / self.t_impr
        else:
            scores = scores / self.t_find
        return scores
 
# --------------------------------------------------------------------------
# 2) Main evaluate_model_rerank function
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model_rerank(
    actor,                # MultiImageActor (possibly DDP)
    critic,               # MultiImageCritic (possibly DDP)
    data_loader,
    tokenizer,
    device,
    chex_scorer,
    cfg,
    *,
    n_candidates: int = 8,
    max_seq_len: int = 512,
    top_p: float = 0.9,
    temperature_find: float = 1.0,
    temperature_impr: float = 0.3,
    test_subset_frac: float = 0.2,
):
    """
    Evaluate with top-p sampling & impression-based temperature scheduling,
    then re-rank n_candidates with the critic. Final parse => Findings vs. Impression.


    Returns a 6-tuple of metrics:
      (cls_f1, BLEU_1, BLEU_4, METEOR, ROUGE_L, full_f1_micro)


    Key points:
      - We encode the images (Swin) exactly once per batch => feats_padded, mask_padded.
      - We feed those features to the actor's GPT2 wrapper (like GPT2GenerationWrapper) for cross-attn.
      - We do n_candidates sampling => shape [B*n_candidates, seq_len].
      - We re-rank by passing partial tokens + feats to the *critic* (no re-running Swin).
      - We parse out "Findings:" / "Impression:" from final best sequence.


    This avoids shape errors in Swin (no "expected 4, got 1").
    """
    start_time = time.perf_counter()
    # Possibly DDP
    ddp_actor = isinstance(actor, nn.parallel.DistributedDataParallel)
    ddp_critic= isinstance(critic, nn.parallel.DistributedDataParallel)
    base_actor  = actor.module  if ddp_actor  else actor
    base_critic = critic.module if ddp_critic else critic
 
    # 0) Prep
    base_actor.eval().to(device)
    base_critic.eval().to(device)
 
    # We assume you have a GPT2GenerationWrapper or similar
    # that lets us do .generate(...) with cross-attention
    gen_wrap = GPT2GenerationWrapper(base_actor).eval().to(device)
 
    # We'll use the ImpressionTemperatureProcessor
    proc = ImpressionTemperatureProcessor(
        tokenizer, temp_find=temperature_find, temp_impr=temperature_impr
    )
    results_per_rank = []
 
    # 1) aggregator classification storage
    cls_pred_list = []
    cls_gt_list   = []
    all_refs, all_gens = [], []
    all_find_refs, all_imp_refs = [], []
 
    # 2) references
    ref_findings_all    = []
    ref_impressions_all = []
 
    # 3) generated
    gen_findings_all    = []
    gen_impressions_all = []
 
    # aggregator ground-truth
    all_chex_gt = []
 
    # We'll prefix with "Findings:"
    prefix_ids = tokenizer.encode("Findings:", add_special_tokens=False)
    prefix_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device)
 
    # subset logic
    total_batches = len(data_loader)
    max_batches   = int(test_subset_frac * total_batches)
    disable_tqdm  = (torch.distributed.is_initialized()
                     and torch.distributed.get_rank() != 0)
 
    # ----------------------------------------------------------------------
    # MAIN LOOP
    # ----------------------------------------------------------------------
    for b_idx, batch in tqdm(enumerate(data_loader), total=max_batches,
                             desc="Eval(ReRank)", disable=disable_tqdm):
        if b_idx >= max_batches:
            break
 
        B = len(batch["report_all"])
        if B == 0:
            continue
 
        # -----------------------------------------------------------
        # 2A) Build image features once for aggregator & critic
        # -----------------------------------------------------------
        # This is the step that calls your Swin encoder exactly once
        feats_padded, mask_padded = build_multi_image_encoder_states(
            batch["images"], base_actor.encoder, device=device, if_frozen=True
        )
 
        # aggregator classification F1
        # aggregator => pooled => cls_head => threshold
        logits = compute_agg_logits(base_actor, feats_padded, mask_padded)
        cls_bin = (torch.sigmoid(logits) >= 0.5).long()
        cls_pred_list.append(cls_bin.cpu())

        # aggregator GT
        if "chexbert_labels" in batch:
            gt_bin = labels_to_bin(batch["chexbert_labels"])
            cls_gt_list.append(gt_bin)
            all_chex_gt.extend(batch["chexbert_labels"])
        else:
            # fallback
            cls_gt_list.append(torch.zeros_like(cls_bin, dtype=torch.long))
            all_chex_gt.extend([[]]*B)
 
        # 2B) references (Findings, Impression)
        gt_findings = batch["findings"]
        gt_impress  = batch["impression"]
        for i in range(B):
            ref_findings_all.append(gt_findings[i])
            ref_impressions_all.append(gt_impress[i])
 
        # -----------------------------------------------------------
        # 3) MULTI-CANDIDATE GENERATION
        # -----------------------------------------------------------
        # We'll replicate feats/mask for n_candidates
        feats_rep = feats_padded.repeat_interleave(n_candidates, dim=0).contiguous()
        mask_rep  = mask_padded.repeat_interleave(n_candidates, dim=0).contiguous()
 
        # set these in your generation wrapper
        gen_wrap.set_encoder_states(feats_rep, mask_rep)
 
        # build input prefix => shape [B*n_candidates, prefix_len]
        prefix_big = prefix_ids.unsqueeze(0).expand(B * n_candidates, -1)
        att_big    = torch.ones_like(prefix_big)
 
        # do top-p sampling => pass our custom processor that changes temperature
        # after <impression>
        gen_out = gen_wrap.generate(
            input_ids=prefix_big,
            attention_mask=att_big,
            max_new_tokens=max_seq_len,
            do_sample=True,
            top_p=top_p,
            logits_processor=[proc],  # switch temperature post-impression
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
        gen_wrap.clear_encoder_states()
 
        # shape [B*n_candidates, seq_len]
        seqs = gen_out.view(B, n_candidates, -1)  # [B, n_candidates, seq_len]
 
        # -----------------------------------------------------------
        # 4) CRITIC RE-RANK
        # -----------------------------------------------------------
        vals_list = []
        for cand_i in range(n_candidates):
            cand_seq = seqs[:, cand_i, :]  # shape [B, seq_len]
            # token-level att
            cand_att = (cand_seq != tokenizer.pad_token_id).long()
            cand_pos = cand_att.cumsum(1) - 1
 
            # pass to critic => already have feats_padded, mask_padded?
            # Actually we want feats for each row => feats_rep but slice
            # because feats_rep shape [B*n_candidates, ...].
            # We'll do the same trick: cand_feats = feats_rep[ cand_i::n_candidates ]
            # but that re-run is more complicated. Instead, let's do direct call:
            # We'll pass feats_padded=feats_rep, then index each row?
            # The simplest is "re-slice" from feats_rep, mask_rep:
            slice_from = cand_i
            cand_feats = feats_rep[slice_from::n_candidates].contiguous()
            cand_mask  = mask_rep [slice_from::n_candidates].contiguous()
 
            # forward => shape [B, seq_len], last hidden => value
            out_critic = base_critic(
                batch_images=None,
                partial_tokens=cand_seq,
                feats_padded=cand_feats,
                mask_padded=cand_mask,
                device=device,
                attention_mask=cand_att,
                position_ids=cand_pos
            )
            # out_critic => shape [B, seq_len]
            # take last position => shape [B]
            cand_val = out_critic[:, -1]
            vals_list.append(cand_val)
 
        # stack => [B, n_candidates]
        vals_stack = torch.stack(vals_list, dim=1)
        best_idx   = vals_stack.argmax(dim=1)  # shape [B]
        best_seqs  = seqs[ torch.arange(B), best_idx, : ]  # shape [B, seq_len]
 
        # parse out
        for i_row, row in enumerate(best_seqs):
            txt = tokenizer.decode(row, skip_special_tokens=True).replace("  ", " ").replace("  ", " ").replace("  ", " ")
            gen_find, gen_impr = parse_findings_impression(txt)
 
            gen_findings_all.append(gen_find)
            gen_impressions_all.append(gen_impr)

            # full_txt = f"Findings: {gen_find} Impression: {gen_impr}"
            all_gens.append(txt)
            ref_full = f"Findings: {gt_findings[i_row]} Impression: {gt_impress[i_row]}"
            all_refs.append(ref_full)
            all_find_refs.append(gt_findings[i_row])
            all_imp_refs.append(gt_impress[i_row])
            results_per_rank.append({
                "subject_id": str(batch["subject_id"][i_row]),
                "study_id"  : str(batch["study_id"][i_row]),
                "split"     : str(batch["split"][i_row]),
                "ref"       : ref_full,
                "gen"       : txt,
            })
 
    # ----------------------------------------------------------------------
    # 5) Compute final metrics
    # ----------------------------------------------------------------------
    # (A) aggregator classification F1
    if cls_pred_list and cls_gt_list:
        cls_pred_cat = torch.cat(cls_pred_list, dim=0)
        cls_gt_cat   = torch.cat(cls_gt_list,   dim=0)
        cls_f1, cls_P, cls_R = compute_pos_only_f1(cls_pred_cat, cls_gt_cat)
    else:
        cls_f1, cls_P, cls_R = 0.0, 0.0, 0.0
 
    # (B) NLG => Findings only
    nlg = compute_batch_nlg_metrics(ref_findings_all, gen_findings_all, num_proc=4)
 
    # -----------------------------------------------------
    # (E) micro / macro Precision·Recall·F1  (CheXbert 기반)
    # -----------------------------------------------------
    logging.info(f"[ReRank] Aggregator-F1={cls_f1:.3f}  P={cls_P:.3f}  R={cls_R:.3f}")
    logging.info(
        f"[ReRank] (Findings NLG) BLEU1={nlg['BLEU_1']:.3f}  BLEU4={nlg['BLEU_4']:.3f}  "
        f"METEOR={nlg['METEOR']:.3f}  ROUGE_L={nlg['ROUGE_L']:.3f}"
    )

    # -------------------------------------------------------------
    #  (NEW) 결과 JSON / YAML 파일 저장  ― one_shot 과 동일 패턴
    # -------------------------------------------------------------
    rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
    if rank0:
        metrics = compute_all_metrics(
            all_refs,
            all_gens,
            ref_findings_all,
            gen_findings_all,
            ref_impressions_all,
            gen_impressions_all,
            chex_scorer,
        )
        elapsed = time.perf_counter() - start_time
        save_results("eval_rerank", results_per_rank, metrics, cfg, elapsed)
    
 
    # Show sample lines
    num_examples = min(2, len(gen_findings_all))
    for i in range(num_examples):
        logging.info(f"--- Example {i} ---")
        logging.info(f"REF-FIND: {ref_findings_all[i]}")
        logging.info(f"GEN-FIND: {gen_findings_all[i]}")
        logging.info(f"REF-IMPR: {ref_impressions_all[i]}")
        logging.info(f"GEN-IMPR: {gen_impressions_all[i]}")
 
    # Return 6-tuple
    return (
        cls_f1,
        metrics["Full_BLEU_1"],
        metrics["Full_BLEU_4"],
        metrics["Full_METEOR"],
        metrics["Full_ROUGE_L"],
        metrics["Full_F1_micro"],
    )

# ---------------------------------------------------------------------
#                   DDP & 로거 설정 헬퍼
# ---------------------------------------------------------------------
def setup_logger(log_dir: str):
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"eval_log_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, "w"),
        ],
    )
    return logging.getLogger()

def setup_distributed(backend="nccl"):
    if int(os.environ.get("RANK", -1)) != -1:
        init_process_group(backend=backend, timeout=timedelta(hours=3))
        rank = int(os.environ["RANK"])
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return True, f"cuda:{local}", rank == 0
    return False, "cuda" if torch.cuda.is_available() else "cpu", True

def cleanup_distributed():
    if torch.distributed.is_initialized():
        destroy_process_group()

# ---------------------------------------------------------------------
#                               main
# ---------------------------------------------------------------------
def main(cfg):
    ddp, device, master = setup_distributed()
    if master:
        setup_logger(cfg.log_dir)

    # ---- tokenizer & special tokens ----
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    specials = []
    for cond in CONDITIONS:
        s = f"<{cond.replace(' ', '_')}>"
        if s not in tok.get_vocab():
            specials.append(s)
    for cls in CLASS_MAPPING.values():
        s = f"<{cls.replace(' ', '_')}>"
        if s not in tok.get_vocab():
            specials.append(s)
    for s in ["<cls_end>","<next>","<impression>"]:
        if s not in tok.get_vocab():
            specials.append(s)
    if specials:
        tok.add_special_tokens({"additional_special_tokens": specials})
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})

    # ---- data ----
    transform=T.Compose([
        T.Resize(cfg.image_size+64),T.CenterCrop(cfg.image_size),
        T.ToTensor(),
        T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    if cfg.dataset_name == "mimiccxr":
        test_ds = MIMICCXRDataset(cfg.mimic_csv_path,transform,"test",
                                cfg.filter_findings,cfg.max_images,cfg.dataset_option, cfg.dataset_name)
        train_ds = MIMICCXRDataset(cfg.mimic_csv_path, transform, "train",
                                  cfg.filter_findings, cfg.max_images, cfg.dataset_option, cfg.dataset_name)
        validation_ds = MIMICCXRDataset(cfg.mimic_csv_path, transform, "validate",
                                       cfg.filter_findings, cfg.max_images, cfg.dataset_option, cfg.dataset_name)
    else:
        test_ds=MIMICCXRDataset(cfg.iu_csv_path,transform,"test",
                                cfg.filter_findings,cfg.max_images,cfg.dataset_option, cfg.dataset_name)
    test_loader=DataLoader(
        test_ds,batch_size=cfg.test_batch_size,shuffle=False,
        collate_fn=my_collate_fn,num_workers=cfg.num_workers,pin_memory=True
    )

    # ---- 모델 ----
    actor = MultiImageActor(len(tok))
    actor.to(device)
    critic = MultiImageCritic()
    critic.to(device)
    critic.gpt2.resize_token_embeddings(len(tok))
    # (필요 시 actor/critic 체크포인트 로드)
    if cfg.ckpt_path:
        load_actor_critic_from_ckpt(cfg.ckpt_path, actor, critic, device)

    base_actor_debug = None

    if ddp:
        actor=DDP(actor,device_ids=[torch.cuda.current_device()],find_unused_parameters=True)
        critic=DDP(critic,device_ids=[torch.cuda.current_device()],find_unused_parameters=True)
        if base_actor_debug is not None:
            base_actor_debug = DDP(
                base_actor_debug,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=True,
            )

    chex=CheXbertScorer(cfg.chexbert_path,device)

    # ---- evaluate ----
    if cfg.eval_mode=="one_shot":
        metrics=evaluate_model(
            actor,
            test_loader,
            tok,
            device,
            chex,
            cfg,
            max_seq_len=cfg.max_seq_len_test,
            test_subset_frac=cfg.eval_only_subset_frac,
        )
    elif cfg.eval_mode == "flex_next":
        metrics = evaluate_model_flexible_next(
            actor,
            critic,
            test_loader,
            tok,
            device,
            chex,
            cfg,
            forced_next_count=cfg.forced_next_count,
            max_seq_len=cfg.max_seq_len_test,
            test_subset_frac=cfg.eval_only_subset_frac,
            impr_n=cfg.impr_n,
            find_top_p=cfg.findings_top_p,
            impr_top_p=cfg.impression_top_p,
            findings_temperature=cfg.findings_temperature,
            impression_temperature=cfg.impression_temperature,
        )
    elif cfg.eval_mode == "flex_next_whole_bon":
        metrics = flex_next_whole_bon(
            actor,
            critic,
            test_loader,
            tok,
            device,
            chex,
            cfg,
            forced_next_count=cfg.forced_next_count,
            max_seq_len=cfg.max_seq_len_test,
            test_subset_frac=cfg.eval_only_subset_frac,
            n_samples=cfg.n_samples,
            find_top_p=cfg.findings_top_p,
            impr_top_p=cfg.impression_top_p,
            find_temp=cfg.findings_temperature,
            impr_temp=cfg.impression_temperature,
        )
    else:
        raise ValueError(cfg.eval_mode)

    if master:
        cls_f1,b1,b4,mt,rg,txt_f1=metrics
        logging.info(f"[Eval-{cfg.eval_mode}] CLS-F1={cls_f1:.3f} "
                     f"BLEU1={b1:.3f} BLEU4={b4:.3f} METEOR={mt:.3f} "
                     f"ROUGE_L={rg:.3f} Text-F1={txt_f1:.3f}")

    cleanup_distributed()

# ---------------------------------------------------------------------
if __name__=="__main__":
    p=argparse.ArgumentParser(description="MIMIC-CXR Report Generation 평가 전용")
    p.add_argument("--mimic_csv_path",type=str,default="YOUR preprocessed MIMIC-CXR data(csv) file")
    p.add_argument("--iu_csv_path",type=str,default="YOUR preprocessed iu-Xray data(csv) file")
    p.add_argument("--chexbert_path",type=str,default="YOUR CheXbert CHECKPOINT PATH")
    p.add_argument("--ckpt_path", type=str,default="YOUR CHECKPOINT PATH")
    p.add_argument("--image_size",type=int,default=512)
    p.add_argument("--dataset_name", type=str,default="iuxray", choices=["mimiccxr", "iuxray"])
    p.add_argument("--dataset_option",type=str,default="findings_impression",
                choices=["findings_only","findings_impression"])
    p.add_argument("--mode",type=str,default="test",choices=["test"])

    # one_shot -> greedy without special token controls
    # flex_next -> + next forcing
    # flex_next_whole_bon -> + next forcing + BoN + T scheduling
    p.add_argument("--eval_mode",type=str,default="one_shot",
                   choices=["one_shot","flex_next","flex_next_whole_bon"])

    # flex next options
    p.add_argument("--forced_next_count", type=int, default=10)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--impression_temperature", type=float, default=1.0)
    p.add_argument("--findings_temperature", type=float, default=1.0)
    p.add_argument("--findings_top_p", type=float, default=0.9)
    p.add_argument("--impression_top_p", type=float, default=0.9)

    p.add_argument("--eval_only_subset_frac",type=float,default=1)
    p.add_argument("--max_seq_len_test",type=int,default=512)

    p.add_argument("--test_batch_size",type=int,default=8)
    p.add_argument("--num_workers",type=int,default=4)
    p.add_argument("--filter_findings",action="store_true",default=True)

    p.add_argument("--max_images",type=int,default=3)
    p.add_argument("--log_dir", type=str, default="YOUR log directory")
    p.add_argument("--seed", type=int, default=43)
    cfg=p.parse_args()

    set_seeds(cfg.seed)
    main(cfg)
