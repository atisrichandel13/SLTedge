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
import gc
import json
import os
import time

import torch
from torch import nn
from transformers import MT5ForConditionalGeneration, T5Tokenizer

from .stgcn_layers import Graph, get_stgcn_chain

PARTS = ["body", "left", "right", "face_all"]


class PoseOnlyUniSign(nn.Module):
    def __init__(self, mt5_path, hidden_dim=256, lang="English", keep_ids=None, device=None):
        """device: where from_pretrained materialises mT5. On the 8 GB unified-memory Jetson the
        full FP32 model (2.3 GB) must land on the GPU directly; a CPU copy plus a GPU copy OOMs."""
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

        load_kw = {"device_map": str(device)} if device is not None and str(device) != "cpu" else {}
        self._mt5_path = mt5_path  # kept so attach_pruned_tokenizer can find keep_ids.json
        self.mt5_model = MT5ForConditionalGeneration.from_pretrained(mt5_path, **load_kw)
        self.mt5_tokenizer = T5Tokenizer.from_pretrained(mt5_path, legacy=False)
        self.prefix = f"Translate sign language video to {self.lang}: "
        self.keep_ids = None
        if keep_ids is not None:
            self.prune_vocab(keep_ids)

    def attach_pruned_tokenizer(self, keep_ids=None):
        """For mT5 weights that are ALREADY pruned on disk (a pre-pruned HF directory).

        prune_and_save() writes a checkpoint file, not an HF directory, and PrunedTokenizer is a
        runtime wrapper with no save_pretrained -- so a pre-pruned directory carries pruned WEIGHTS
        next to the ORIGINAL 250k tokenizer. Loading it without this call produces fluent but wrong
        text, because every token id is off. This does the non-weight half of prune_vocab: the
        tokenizer remap, the generation ids, and keep_ids.

        Refuses to run if the weights are not in fact already pruned, so a wrong assumption about the
        directory fails loudly instead of silently mistranslating.
        """
        if keep_ids is None:
            # The remap data ships inside a pre-pruned directory, so read it from there rather than
            # making every caller pass it. Hard-fail if absent: a pre-pruned directory whose keep set
            # we cannot recover is unusable, because spiece.model is the ORIGINAL 250k sentencepiece
            # (byte-identical to google/mt5-base's) and every id would be wrong with no error raised.
            kp = os.path.join(self._mt5_path, "keep_ids.json")
            if not os.path.exists(kp):
                raise RuntimeError(
                    f"attach_pruned_tokenizer: no keep_ids passed and no keep_ids.json in "
                    f"{self._mt5_path}. Refusing to proceed: the directory's tokenizer is the "
                    f"original 250k vocabulary, so without the remap every token id is wrong and the "
                    f"model would emit fluent, incorrect text silently.")
            with open(kp) as f:
                keep_ids = json.load(f)
            if isinstance(keep_ids, dict):
                keep_ids = keep_ids.get("keep_ids", keep_ids)
        keep = torch.as_tensor(sorted(int(i) for i in keep_ids), dtype=torch.long)
        assert keep[:3].tolist() == [0, 1, 2], "keep_ids must contain pad/eos/unk = 0/1/2"
        m = self.mt5_model
        have = m.shared.weight.data.shape[0]
        if have != len(keep):
            raise RuntimeError(
                f"attach_pruned_tokenizer: mT5 embedding has {have} rows but keep_ids has "
                f"{len(keep)}. This directory is not pre-pruned to match the checkpoint; use "
                f"prune_vocab() instead. (The 26,078-id keep set was superseded by 26,025 on "
                f"2026-09-30 when the test-set leak was fixed, so a stale directory lands here.)")
        m.config.vocab_size = len(keep)
        if getattr(m, "generation_config", None) is not None:
            m.generation_config.pad_token_id, m.generation_config.eos_token_id = 0, 1
            m.generation_config.decoder_start_token_id = 0
        self.keep_ids = keep
        self.mt5_tokenizer = PrunedTokenizer(self.mt5_tokenizer, keep)

    # ------------------------------------------------------------------ vocabulary pruning (L5)
    def prune_vocab(self, keep_ids):
        """Slice mT5's input embedding and LM head to `keep_ids` (sorted old token ids).
        New id = position in keep_ids. pad/eos/unk (0/1/2) must be kept so decoder_start (0) and
        eos (1) survive unchanged. The tokenizer is wrapped to remap ids both ways."""
        keep = torch.as_tensor(sorted(int(i) for i in keep_ids), dtype=torch.long)
        assert keep[:3].tolist() == [0, 1, 2], "keep_ids must contain pad/eos/unk = 0/1/2"
        m = self.mt5_model
        old_shared, old_head = m.shared.weight.data, m.lm_head.weight.data
        keep_dev = keep.to(old_shared.device)  # weights may already be on the GPU
        new_shared = nn.Embedding(len(keep), old_shared.shape[1])
        new_shared.weight.data = old_shared[keep_dev].clone()
        m.set_input_embeddings(new_shared)          # also rewires encoder/decoder embed_tokens
        new_head = nn.Linear(old_head.shape[1], len(keep), bias=False)
        new_head.weight.data = old_head[keep_dev].clone()
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
        # old2new/keep are CPU lookup tables built at prune time, but `ids` can arrive on CUDA
        # (generate() returns device tensors). Index on the table's device, return on the
        # caller's. CPU-only runs never hit this; the first GPU eval did.
        ids = torch.as_tensor(ids, dtype=torch.long)
        dev = ids.device
        ids_c = ids.to(self.old2new.device)
        out = torch.full_like(ids_c, 2)
        ok = ids_c < len(self.old2new)
        out[ok] = self.old2new[ids_c[ok]]
        return out.to(dev)

    def __call__(self, text, **kw):
        enc = self.tok(text, **kw)
        enc["input_ids"] = self._map(enc["input_ids"]) if kw.get("return_tensors") == "pt" \
            else ([self._map(x).tolist() for x in enc["input_ids"]] if isinstance(text, list) else self._map(enc["input_ids"]).tolist())
        return enc

    def decode(self, ids, **kw):
        # index on the lookup table's own device and return on the caller's: generate() hands back
        # ids on CUDA while keep/old2new are CPU tables built at prune time
        ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1).to(self.keep.device)
        return self.tok.decode(self.keep[ids].tolist(), **kw)

    def batch_decode(self, seqs, **kw):
        return [self.decode(s, **kw) for s in seqs]


def _mt5_vocab_size(mt5_path):
    """vocab_size from the directory's config.json, without loading any weights. None if unknown."""
    try:
        with open(os.path.join(mt5_path, "config.json")) as f:
            return int(json.load(f)["vocab_size"])
    except Exception:  # noqa: BLE001 -- a hub id, a missing file, anything: fall back to slicing
        return None


def load_model(ckpt_path, mt5_path, device="cpu", dtype=torch.float32, keep_ids=None,
               w8_runtime="dequant"):
    """keep_ids: list of old token ids -> build the model, load the full checkpoint, then prune.
    A checkpoint saved by prune_and_save() carries its own keep_ids and is loaded pruned.
    A checkpoint saved by unisign.quant carries w8_keys; w8_runtime = "dequant" (float weights in
    RAM) or "int8" (W8Linear modules, int8 in RAM, dequantised per forward)."""
    # mmap: the 2.3 GB full checkpoint stays page-cache backed instead of anonymous RAM, which
    # matters on the 8 GB unified-memory Jetson where CPU + GPU copies share one pool.
    sd = torch.load(ckpt_path, map_location="cpu", mmap=True)
    ckpt_keep = sd.get("keep_ids", None)
    w8_keys = sd.get("w8_keys", None)
    sd = sd.get("model", sd)
    if w8_keys:
        from .quant import dequantize_state_dict, swap_linears_to_w8
        q_sd = sd
        sd = dequantize_state_dict(sd, w8_keys)
    if ckpt_keep is not None:
        # If mt5_path is ALREADY pruned to this vocabulary, materialise it straight onto the device and
        # only remap the tokenizer. The alternative below builds the full 250k-vocab mT5 first and
        # slices it, which on this unified-memory board peaks near 4.2 GB (2.3 full + 0.95 sliced +
        # 0.95 device copy) -- more than loading the FULL checkpoint costs -- and fails whenever
        # MemFree cannot reach ~5 GB, which it often cannot (MemAvailable caps around 5.3 GB).
        if _mt5_vocab_size(mt5_path) == len(ckpt_keep):
            model = PoseOnlyUniSign(mt5_path, device=device)
            # pass the CHECKPOINT's keep_ids, and let attach_pruned_tokenizer cross-check them against
            # the directory's keep_ids.json. Matching counts are not enough -- two different keep sets
            # of the same size would remap every id wrongly and emit fluent nonsense without error.
            model.attach_pruned_tokenizer(ckpt_keep)
            _dir_kp = os.path.join(mt5_path, "keep_ids.json")
            if os.path.exists(_dir_kp):
                with open(_dir_kp) as f:
                    _d = json.load(f)
                _d = _d.get("keep_ids", _d) if isinstance(_d, dict) else _d
                if sorted(int(i) for i in _d) != sorted(int(i) for i in ckpt_keep):
                    raise RuntimeError(
                        f"{mt5_path}/keep_ids.json disagrees with the checkpoint's keep_ids "
                        f"({len(_d)} vs {len(ckpt_keep)} ids, or same count but different ids). "
                        f"These must be the same vocabulary or every token id is wrong.")
        else:
            # pruned: build on CPU and slice first (peak 1.0 GB on the GPU instead of 2.6 GB)
            model = PoseOnlyUniSign(mt5_path, keep_ids=ckpt_keep)
    else:
        # full: materialise mT5 on the target device; CPU copy + GPU copy OOMs the Jetson
        model = PoseOnlyUniSign(mt5_path, device=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # strict=False only to give a readable report; anything missing except mt5 buffers is a bug
    bad_missing = [k for k in missing if not k.startswith("mt5_model.")]
    if bad_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={bad_missing[:5]} unexpected={unexpected[:5]}")
    if keep_ids is not None and ckpt_keep is None:
        model.prune_vocab(keep_ids)
    del sd, missing, unexpected  # free the checkpoint before the device copy (OOMed on the Jetson)
    gc.collect()
    model.eval().to(device=device, dtype=dtype)
    if w8_keys and w8_runtime == "int8":
        n = swap_linears_to_w8(model, q_sd, w8_keys)
        print(f"[load] W8A32 runtime: {n} Linear modules hold int8 weights")
    model.w8 = bool(w8_keys)
    return model


def prune_and_save(ckpt_path, mt5_path, keep_ids, out_path):
    """Full checkpoint -> pruned checkpoint file {"model": state_dict, "keep_ids": [...]}."""
    model = load_model(ckpt_path, mt5_path, keep_ids=keep_ids)
    torch.save({"model": model.state_dict(), "keep_ids": model.keep_ids.tolist()}, out_path)
    return model
