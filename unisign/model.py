"""Pose-only Uni-Sign, standalone. Interface C5 in WORKSPLIT.md.

A faithful subset of `models.py: Uni_Sign` from github.com/ZechengLi19/Uni-Sign (commit eed438b),
keeping only the pose path so the released `*_pose_only_slt.pth` checkpoints load with strict=True.
Dropped: the RGB branch (EfficientNet-B0, deformable attention, fusion gate), deepspeed, decord,
the dataset/config modules. Attribute names are unchanged on purpose: the state dict must match.

Dependencies: torch, transformers (>=4.40), sentencepiece, numpy.

Architecture (pose-only):
    per part in [body, left, right, face_all]:
        Linear(3 -> 64) -> spatial ST-GCN chain (64 -> 256) -> [+ body wrist/nose feature for
        hands/face] -> temporal ST-GCN chain (256, kernel 5) -> mean over joints  => (B, T, 256)
    concat parts (B, T, 1024) + part_para -> Linear(1024 -> 768) => pose tokens
    prefix "Translate sign language video to English: " token embeddings ++ pose tokens -> mT5-base
Note: left and right hand share weights (the repo aliases the modules), so 3 of everything, not 4.
"""
import time

import torch
from torch import nn
from transformers import MT5ForConditionalGeneration, T5Tokenizer

from .stgcn_layers import Graph, get_stgcn_chain

PARTS = ["body", "left", "right", "face_all"]


class PoseOnlyUniSign(nn.Module):
    def __init__(self, mt5_path, hidden_dim=256, lang="English", keep_ids=None):
        super().__init__()
        self.modes = PARTS
        self.lang = lang
        self.graph, A = {}, []
        self.proj_linear = nn.ModuleDict()
        for mode in self.modes:
            self.graph[mode] = Graph(layout=mode, strategy="distance", max_hop=1)
            A.append(torch.tensor(self.graph[mode].A, dtype=torch.float32, requires_grad=False))
            self.proj_linear[mode] = nn.Linear(3, 64)
        self.gcn_modules = nn.ModuleDict()
        self.fusion_gcn_modules = nn.ModuleDict()
        spatial_kernel_size = A[0].size(0)
        for index, mode in enumerate(self.modes):
            self.gcn_modules[mode], final_dim = get_stgcn_chain(64, "spatial", (1, spatial_kernel_size), A[index].clone(), True)
            self.fusion_gcn_modules[mode], _ = get_stgcn_chain(final_dim, "temporal", (5, spatial_kernel_size), A[index].clone(), True)
        # weight sharing between hands, exactly as in the repo
        self.gcn_modules["left"] = self.gcn_modules["right"]
        self.fusion_gcn_modules["left"] = self.fusion_gcn_modules["right"]
        self.proj_linear["left"] = self.proj_linear["right"]

        self.part_para = nn.Parameter(torch.zeros(hidden_dim * len(self.modes)))
        self.pose_proj = nn.Linear(256 * 4, 768)

        self.mt5_model = MT5ForConditionalGeneration.from_pretrained(mt5_path)
        self.mt5_tokenizer = T5Tokenizer.from_pretrained(mt5_path, legacy=False)
        self.prefix = f"Translate sign language video to {self.lang}: "
        self.keep_ids = None
        if keep_ids is not None:
            self.prune_vocab(keep_ids)

    # ------------------------------------------------------------------ vocabulary pruning (L5)
    def prune_vocab(self, keep_ids):
        """Slice mT5's input embedding and LM head to `keep_ids` (sorted old token ids).
        New id = position in keep_ids. pad/eos/unk (0/1/2) must be kept so decoder_start (0) and
        eos (1) survive unchanged. The tokenizer is wrapped to remap ids both ways."""
        keep = torch.as_tensor(sorted(int(i) for i in keep_ids), dtype=torch.long)
        assert keep[:3].tolist() == [0, 1, 2], "keep_ids must contain pad/eos/unk = 0/1/2"
        m = self.mt5_model
        old_shared, old_head = m.shared.weight.data, m.lm_head.weight.data
        new_shared = nn.Embedding(len(keep), old_shared.shape[1])
        new_shared.weight.data = old_shared[keep].clone()
        m.set_input_embeddings(new_shared)          # also rewires encoder/decoder embed_tokens
        new_head = nn.Linear(old_head.shape[1], len(keep), bias=False)
        new_head.weight.data = old_head[keep].clone()
        m.lm_head = new_head
        m.config.vocab_size = len(keep)
        if getattr(m, "generation_config", None) is not None:
            m.generation_config.pad_token_id, m.generation_config.eos_token_id = 0, 1
            m.generation_config.decoder_start_token_id = 0
        self.keep_ids = keep
        self.mt5_tokenizer = PrunedTokenizer(self.mt5_tokenizer, keep)

    # ------------------------------------------------------------------ pose stack
    def encode_pose(self, src_input):
        """src_input: dict from common.pose_to_unisign.collate. Returns (B, T, 768) pose tokens."""
        features = []
        body_feat = None
        for part in self.modes:
            proj_feat = self.proj_linear[part](src_input[part]).permute(0, 3, 1, 2)  # B,C,T,V
            gcn_feat = self.gcn_modules[part](proj_feat)
            if part == "body":
                body_feat = gcn_feat
            elif part == "left":
                gcn_feat = gcn_feat + body_feat[..., -2][..., None].detach()
            elif part == "right":
                gcn_feat = gcn_feat + body_feat[..., -1][..., None].detach()
            elif part == "face_all":
                gcn_feat = gcn_feat + body_feat[..., 0][..., None].detach()
            gcn_feat = self.fusion_gcn_modules[part](gcn_feat)
            features.append(gcn_feat.mean(-1).transpose(1, 2))  # B,T,C
        inputs_embeds = torch.cat(features, dim=-1) + self.part_para
        return self.pose_proj(inputs_embeds)

    def build_encoder_inputs(self, src_input):
        pose_embeds = self.encode_pose(src_input)
        B = pose_embeds.shape[0]
        prefix_token = self.mt5_tokenizer([self.prefix] * B, padding="longest", truncation=True,
                                          return_tensors="pt").to(pose_embeds.device)
        prefix_embeds = self.mt5_model.encoder.embed_tokens(prefix_token["input_ids"])
        inputs_embeds = torch.cat([prefix_embeds, pose_embeds], dim=1)
        attention_mask = torch.cat([prefix_token["attention_mask"],
                                    src_input["attention_mask"].to(pose_embeds.device)], dim=1)
        return inputs_embeds, attention_mask

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def translate(self, src_input, max_new_tokens=64, num_beams=1, timing=None):
        """Returns dict(text, tokens, token_logprobs, timing_ms). Greedy when num_beams == 1."""
        dev = next(self.parameters()).device
        src_input = {k: (v.to(dev).float() if torch.is_tensor(v) and v.is_floating_point() else v)
                     for k, v in src_input.items()}
        sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
        t = {}
        sync(); t0 = time.perf_counter()
        inputs_embeds, attention_mask = self.build_encoder_inputs(src_input)
        sync(); t["gcn"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        enc = self.mt5_model.get_encoder()(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
        sync(); t["encoder"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        out = self.mt5_model.generate(encoder_outputs=enc, attention_mask=attention_mask,
                                      max_new_tokens=max_new_tokens, num_beams=num_beams,
                                      output_scores=True, return_dict_in_generate=True)
        sync(); t["decoder"] = (time.perf_counter() - t0) * 1000

        seq = out.sequences[0]
        tokens = seq[1:].tolist()  # drop decoder start token
        logprobs = []
        if num_beams == 1 and out.scores:
            for step, s in enumerate(out.scores):
                if step >= len(tokens):
                    break
                logprobs.append(float(torch.log_softmax(s[0].float(), -1)[tokens[step]]))
        text = self.mt5_tokenizer.decode(seq, skip_special_tokens=True)
        t["total"] = t["gcn"] + t["encoder"] + t["decoder"]
        t["encoder_len"] = int(attention_mask.shape[1])
        t["decoded_tokens"] = len(tokens)
        return {"text": text, "tokens": tokens, "token_logprobs": logprobs, "timing_ms": t}


class PrunedTokenizer:
    """Wraps the original T5Tokenizer: encode with the full vocab, then map old ids -> new ids
    (unseen tokens -> unk=2); decode maps new -> old then uses the original decoder."""

    def __init__(self, tok, keep):
        self.tok, self.keep = tok, keep
        self.old2new = torch.full((int(keep.max()) + 1,), 2, dtype=torch.long)
        self.old2new[keep] = torch.arange(len(keep))
        self.pad_token_id, self.eos_token_id, self.unk_token_id = 0, 1, 2

    def __len__(self):
        return len(self.keep)

    def _map(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long)
        out = torch.full_like(ids, 2)
        ok = ids < len(self.old2new)
        out[ok] = self.old2new[ids[ok]]
        return out

    def __call__(self, text, **kw):
        enc = self.tok(text, **kw)
        enc["input_ids"] = self._map(enc["input_ids"]) if kw.get("return_tensors") == "pt" \
            else ([self._map(x).tolist() for x in enc["input_ids"]] if isinstance(text, list) else self._map(enc["input_ids"]).tolist())
        return enc

    def decode(self, ids, **kw):
        ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
        return self.tok.decode(self.keep[ids].tolist(), **kw)

    def batch_decode(self, seqs, **kw):
        return [self.decode(s, **kw) for s in seqs]


def load_model(ckpt_path, mt5_path, device="cpu", dtype=torch.float32, keep_ids=None):
    """keep_ids: list of old token ids -> build the model, load the full checkpoint, then prune.
    A checkpoint saved by prune_and_save() carries its own keep_ids and is loaded pruned."""
    sd = torch.load(ckpt_path, map_location="cpu")
    ckpt_keep = sd.get("keep_ids", None)
    sd = sd.get("model", sd)
    if ckpt_keep is not None:
        model = PoseOnlyUniSign(mt5_path, keep_ids=ckpt_keep)
    else:
        model = PoseOnlyUniSign(mt5_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # strict=False only to give a readable report; anything missing except mt5 buffers is a bug
    bad_missing = [k for k in missing if not k.startswith("mt5_model.")]
    if bad_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={bad_missing[:5]} unexpected={unexpected[:5]}")
    if keep_ids is not None and ckpt_keep is None:
        model.prune_vocab(keep_ids)
    model.eval().to(device=device, dtype=dtype)
    return model


def prune_and_save(ckpt_path, mt5_path, keep_ids, out_path):
    """Full checkpoint -> pruned checkpoint file {"model": state_dict, "keep_ids": [...]}."""
    model = load_model(ckpt_path, mt5_path, keep_ids=keep_ids)
    torch.save({"model": model.state_dict(), "keep_ids": model.keep_ids.tolist()}, out_path)
    return model
