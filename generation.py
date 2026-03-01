import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from config import Config as config
from data_preprocess.tokenizer import BPETokenizer, get_tokenizer
from model import NGCTransformer


DEFAULT_MODEL_NAME = "ngc_transformer"
DEFAULT_BPE_TOKENIZER_PATH = Path("data_preprocess") / "outputs" / "tokenizer" / "bpe_tokenizer.json"


def build_model(load_dir: str, model_name: str) -> NGCTransformer:
    dkey = jax.random.PRNGKey(config.SEED)
    return NGCTransformer(
        dkey=dkey,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        n_embed=config.n_embed,
        vocab_size=config.vocab_size,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        T=config.n_iter,
        dt=1.0,
        tau_m=config.tau_m,
        act_fx=config.act_fx,
        eta=config.eta,
        dropout_rate=config.dropout_rate,
        exp_dir=config.exp_dir,
        model_name=model_name,
        loadDir=load_dir,
        pos_learnable=config.pos_learnable,
        optim_type=config.optim_type,
        wub=config.wub,
        wlb=config.wlb,
    )


def build_tokenizer():
    tokenizer = get_tokenizer(config)

    if isinstance(tokenizer, BPETokenizer):
        tokenizer_file = config.tokenizer_vocab_file
        if tokenizer_file is None:
            tokenizer_file = DEFAULT_BPE_TOKENIZER_PATH
        tokenizer_file = Path(tokenizer_file)
        if not tokenizer_file.exists():
            raise FileNotFoundError(
                f"BPE tokenizer file not found: {tokenizer_file}. "
                "Run `python -m data_preprocess.tokenizer` first or set config.tokenizer_vocab_file."
            )
        tokenizer.load_tokenizer(str(tokenizer_file))

    return tokenizer


def _pick_next_token(next_logits: jnp.ndarray, temperature: float, key):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    scaled_logits = next_logits / temperature
    if key is None:
        return int(jnp.argmax(scaled_logits)), key

    probs = jax.nn.softmax(scaled_logits)
    key, subkey = jax.random.split(key)
    token = int(jax.random.choice(subkey, a=scaled_logits.shape[0], p=probs))
    return token, key


def generate_text(
    model: NGCTransformer,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    seq_len: int = None,
    temperature: float = 1.0,
    key=None,
) -> str:
    if seq_len is None:
        seq_len = config.seq_len

    prompt_ids = tokenizer.encode(prompt)
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if len(prompt_ids) == 0:
        raise ValueError("Prompt is empty after tokenization; provide a non-empty prompt.")

    current_tokens = jnp.array([prompt_ids], dtype=jnp.int32)  # (1, T)
    dummy_target = jnp.zeros((config.batch_size * config.seq_len, config.vocab_size), dtype=jnp.float32)

    for _ in range(max_new_tokens):
        input_seq = current_tokens[:, -seq_len:]
        actual_len = int(input_seq.shape[1])

        if actual_len < seq_len:
            pad_len = seq_len - actual_len
            input_seq = jnp.pad(input_seq, ((0, 0), (0, pad_len)), constant_values=0)

        # Model state size is tied to config.batch_size; duplicate prompt across batch.
        model_input = jnp.repeat(input_seq, repeats=config.batch_size, axis=0)

        _, y_mu, _ = model.process(obs=model_input, lab=dummy_target, adapt_synapses=False)
        logits = y_mu.reshape(config.batch_size, config.seq_len, config.vocab_size)

        last_pos = min(int(current_tokens.shape[1]), seq_len) - 1
        next_logits = logits[0, last_pos, :]
        next_token, key = _pick_next_token(next_logits, temperature=temperature, key=key)

        next_token_arr = jnp.array([[next_token]], dtype=jnp.int32)
        current_tokens = jnp.concatenate([current_tokens, next_token_arr], axis=1)

    return tokenizer.decode(current_tokens[0])


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained NGC-PC-Transformer checkpoint.")
    parser.add_argument("--prompt", type=str, default="The king said: ", help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=120, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (> 0).")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (ignored with --greedy).")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling.")
    parser.add_argument("--checkpoint-dir", type=str, default=config.exp_dir, help="Model checkpoint directory.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Saved model name.")
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}. Run `python train.py` first."
        )

    tokenizer = build_tokenizer()
    model = build_model(load_dir=str(ckpt_dir), model_name=args.model_name)

    key = None if args.greedy else jax.random.PRNGKey(args.seed)
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        seq_len=config.seq_len,
        temperature=args.temperature,
        key=key,
    )
    print(generated)


if __name__ == "__main__":
    main()
