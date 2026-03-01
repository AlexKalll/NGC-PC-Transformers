import argparse
import traceback

import jax
import jax.numpy as jnp

from config import Config as config
from generation import generate_text
from model import NGCTransformer


class SmokeTokenizer:
    """Minimal tokenizer for generation smoke checks."""

    def encode(self, text: str):
        if not text:
            return [0]
        return [ord(ch) % config.vocab_size for ch in text]

    def decode(self, tokens):
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return " ".join(str(int(t)) for t in tokens)


def build_smoke_model(t_steps: int) -> NGCTransformer:
    dkey = jax.random.PRNGKey(config.SEED)
    return NGCTransformer(
        dkey=dkey,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        n_embed=config.n_embed,
        vocab_size=config.vocab_size,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        T=t_steps,
        dt=1.0,
        tau_m=config.tau_m,
        act_fx=config.act_fx,
        eta=config.eta,
        dropout_rate=config.dropout_rate,
        exp_dir=None,
        model_name="smoke_ngc_transformer",
        loadDir=None,
        pos_learnable=config.pos_learnable,
        optim_type=config.optim_type,
        wub=config.wub,
        wlb=config.wlb,
    )


def run_forward_smoke(model: NGCTransformer):
    key = jax.random.PRNGKey(7)
    inputs = jax.random.randint(
        key,
        shape=(config.batch_size, config.seq_len),
        minval=0,
        maxval=config.vocab_size,
        dtype=jnp.int32,
    )
    target_ids = jax.random.randint(
        jax.random.PRNGKey(9),
        shape=(config.batch_size, config.seq_len),
        minval=0,
        maxval=config.vocab_size,
        dtype=jnp.int32,
    )
    targets = jax.nn.one_hot(target_ids.reshape(-1), config.vocab_size, dtype=jnp.float32)

    y_mu_inf, y_mu, efe = model.process(obs=inputs, lab=targets, adapt_synapses=False)

    expected_shape = (config.batch_size * config.seq_len, config.vocab_size)
    assert y_mu_inf.shape == expected_shape, f"Unexpected y_mu_inf shape: {y_mu_inf.shape}"
    assert y_mu.shape == expected_shape, f"Unexpected y_mu shape: {y_mu.shape}"
    assert jnp.isfinite(y_mu_inf).all(), "y_mu_inf contains non-finite values"
    assert jnp.isfinite(y_mu).all(), "y_mu contains non-finite values"
    assert jnp.isfinite(efe), "EFE is non-finite"


def run_generation_smoke(model: NGCTransformer, max_new_tokens: int):
    tokenizer = SmokeTokenizer()
    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="Smoke",
        max_new_tokens=max_new_tokens,
        seq_len=config.seq_len,
        temperature=1.0,
        key=jax.random.PRNGKey(11),
    )
    assert isinstance(text, str), "Generated output must be a string"
    assert len(text) > 0, "Generated output is empty"


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal smoke checks for NGC-PC-Transformers.")
    parser.add_argument("--t-steps", type=int, default=1, help="Internal model steps (default: 1).")
    parser.add_argument("--max-new-tokens", type=int, default=2, help="Tokens for generation smoke.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        model = build_smoke_model(t_steps=args.t_steps)
        run_forward_smoke(model)
        run_generation_smoke(model, max_new_tokens=args.max_new_tokens)
        print("SMOKE OK: forward + generation")
    except Exception as exc:
        print(f"SMOKE FAILED: {exc}")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
