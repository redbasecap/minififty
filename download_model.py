"""Download Mini-LLM (50M params) and save it locally in Hugging Face format.

Same conversion as the Colab notebook: the original training checkpoint is
remapped onto a LlamaForCausalLM, then written to ./model so the server can
start without touching the network.
"""
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

MODEL_NAME = "Ashx098/Mini-LLM"
# The checkpoint on main was replaced by a five-step training run; pin the good one.
REVISION = "6b16cfbd089ec8e09751959c83e622627c24cf7e"
OUT_DIR = Path(__file__).parent / "model"


class ModelArgs:
    """Plain container for the configuration saved by the original trainer."""


def convert():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder="Tokenizer/BPE", token=False)
    path = hf_hub_download(
        MODEL_NAME, "phase-1-pretraining/checkpoints/ckpt.pt", revision=REVISION, token=False
    )
    with torch.serialization.safe_globals([(ModelArgs, "model.config.ModelArgs")]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    cfg = vars(checkpoint["model_args"])
    raw = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()}

    config = LlamaConfig(
        vocab_size=cfg["vocab_size"], hidden_size=cfg["dim"],
        intermediate_size=raw["layers.0.feed_forward.w1.weight"].shape[0],
        num_hidden_layers=cfg["n_layers"], num_attention_heads=cfg["n_heads"],
        num_key_value_heads=cfg["n_kv_heads"], max_position_embeddings=cfg["max_seq_len"],
        rms_norm_eps=1e-6, tie_word_embeddings=True,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(config)
    weights = {
        "model.embed_tokens.weight": raw.pop("tok_embeddings.emb.weight"),
        "model.norm.weight": raw.pop("norm.weight"),
        "lm_head.weight": raw.pop("output.weight"),
    }
    raw.pop("rope.freqs_cis", None)  # transformers derives its own rotary frequencies
    layer_names = {
        "attention.wq": "self_attn.q_proj", "attention.wk": "self_attn.k_proj",
        "attention.wv": "self_attn.v_proj", "attention.wo": "self_attn.o_proj",
        "feed_forward.w1": "mlp.gate_proj", "feed_forward.w2": "mlp.down_proj",
        "feed_forward.w3": "mlp.up_proj", "attention_norm": "input_layernorm",
        "ffn_norm": "post_attention_layernorm",
    }
    for i in range(cfg["n_layers"]):
        for source, target in layer_names.items():
            weight = raw.pop(f"layers.{i}.{source}.weight")
            if source in ("attention.wq", "attention.wk"):
                # Mini-LLM interleaves real/imaginary pairs; Llama groups the halves.
                heads = cfg["n_heads"] if source == "attention.wq" else cfg["n_kv_heads"]
                weight = weight.reshape(heads, -1, 2, cfg["dim"]).transpose(1, 2).reshape_as(weight)
            weights[f"model.layers.{i}.{target}.weight"] = weight
    if raw:
        raise ValueError(f"Unexpected checkpoint tensors: {list(raw)}")
    model.load_state_dict(weights, strict=True)
    return tokenizer, model.eval()


if __name__ == "__main__":
    tokenizer, model = convert()
    OUT_DIR.mkdir(exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Saved {sum(p.numel() for p in model.parameters()):,} parameters to {OUT_DIR}")
