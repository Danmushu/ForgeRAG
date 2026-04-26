"""
Interactive setup wizard for ForgeRAG.

Walks the user through six small steps and writes a forgerag.yaml
that wires the relational store, vector store, blob storage,
embedder, and answer-generation LLM end-to-end. The embedder and
LLM steps each finish with a real connection test (live API call)
so a typo in api_base or a wrong key surfaces immediately, before
the user discovers it the first time they try to ingest a document.

Navigation:
    Enter          accept the default in [yellow]
    b / back / <   re-open the previous step
    Ctrl-C         abort

Usage:
    python scripts/setup.py                       # interactive wizard
    python scripts/setup.py --profile dev -y      # accept dev defaults
    python scripts/setup.py --profile prod -o myconfig.yaml

Profiles:
    dev    -- ChromaDB + local blob + OpenAI defaults
    prod   -- pgvector + local blob + OpenAI defaults
    custom -- full wizard, no presets
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Force UTF-8 on stdout/stderr so the box-drawing / arrow characters in
# banners render on Windows consoles where the default codepage is GBK
# (otherwise UnicodeEncodeError aborts the wizard mid-prompt).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError, OSError):
        pass

# Let the wizard run from the repo root without install.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Optional backend → required package (import_name, pip_name)
# ---------------------------------------------------------------------------

_RELATIONAL_PACKAGES: dict[str, tuple[str, str]] = {
    "postgres": ("psycopg", "psycopg[binary]"),
}

_VECTOR_PACKAGES: dict[str, tuple[str, str]] = {
    "chromadb": ("chromadb", "chromadb"),
    "qdrant": ("qdrant_client", "qdrant-client"),
    "milvus": ("pymilvus", "pymilvus"),
    "weaviate": ("weaviate", "weaviate-client"),
    "pgvector": ("psycopg", "psycopg[binary]"),
}

_BLOB_PACKAGES: dict[str, tuple[str, str]] = {
    "s3": ("boto3", "boto3"),
    "oss": ("oss2", "oss2"),
}


def _ensure_package(import_name: str, pip_name: str) -> None:
    """Install *pip_name* if *import_name* cannot be imported."""
    if importlib.util.find_spec(import_name) is not None:
        return
    print(_c(f"  '{pip_name}' is not installed — installing now…", "yellow"))
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name],
            check=True,
        )
        print(_c(f"  '{pip_name}' installed successfully.", "green"))
    except subprocess.CalledProcessError as exc:
        print(_c(f"  failed to install '{pip_name}': {exc}", "magenta"))
        print(_c(f"  install it manually and re-run: pip install {pip_name}", "dim"))


def _ensure_backend_package(mapping: dict[str, tuple[str, str]], backend: str) -> None:
    """Look up *backend* in *mapping* and install its package if missing."""
    if backend not in mapping:
        return
    import_name, pip_name = mapping[backend]
    _ensure_package(import_name, pip_name)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def _is_tty() -> bool:
    return sys.stdout.isatty()


def _c(text: str, color: str) -> str:
    if not _is_tty():
        return text
    codes = {
        "bold": "1",
        "dim": "2",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
    }
    code = codes.get(color, "0")
    return f"\033[{code}m{text}\033[0m"


def banner(title: str) -> None:
    print()
    print(_c("━" * 60, "dim"))
    print(_c(f" {title}", "bold"))
    print(_c("━" * 60, "dim"))


def section(title: str) -> None:
    print()
    print(_c(f"▸ {title}", "cyan"))


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


class Aborted(Exception):
    """Ctrl-C / EOF — stop the whole wizard."""


class _GoBack(Exception):
    """User typed 'b' / 'back' / '<' — re-run the previous step."""


# Tokens that trigger _GoBack when entered at any prompt.
_BACK_TOKENS = {"b", "back", "<"}


def _check_back(raw: str) -> None:
    if raw.lower() in _BACK_TOKENS:
        raise _GoBack()


def ask(
    question: str,
    default: str | None = None,
    *,
    validator: Callable[[str], str | None] | None = None,
    allow_empty: bool = False,
) -> str:
    """
    Ask for free-form text input. Returns the value (or the default).
    `validator` may return an error message to force a retry. Typing
    ``b`` / ``back`` / ``<`` raises ``_GoBack``.
    """
    suffix = f" [{_c(default, 'yellow')}]" if default else ""
    while True:
        try:
            raw = input(f"  {question}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise Aborted()
        _check_back(raw)
        if not raw and default is not None:
            raw = default
        if not raw and not allow_empty:
            print(_c("  (required)", "magenta"))
            continue
        if validator:
            err = validator(raw)
            if err:
                print(_c(f"  {err}", "magenta"))
                continue
        return raw


def ask_bool(question: str, default: bool = False) -> bool:
    tip = "Y/n" if default else "y/N"
    while True:
        try:
            raw = input(f"  {question} [{_c(tip, 'yellow')}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise Aborted()
        _check_back(raw)
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print(_c("  please answer y or n", "magenta"))


def ask_choice(
    question: str,
    options: list[tuple[str, str]],  # (value, description)
    default: str | None = None,
) -> str:
    """Numbered menu selection. `options[i]` is (value, description)."""
    print(f"  {question}")
    default_idx = None
    for i, (value, desc) in enumerate(options, 1):
        marker = ""
        if default == value:
            default_idx = i
            marker = _c(" (default)", "dim")
        print(f"    {i}) {_c(value, 'bold')}  {_c('— ' + desc, 'dim')}{marker}")
    default_str = str(default_idx) if default_idx else None
    while True:
        try:
            raw = input(
                f"  enter [1-{len(options)}]{' [' + _c(default_str, 'yellow') + ']' if default_str else ''}: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            raise Aborted()
        _check_back(raw)
        if not raw and default_str:
            raw = default_str
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        print(_c(f"  please enter a number 1-{len(options)}", "magenta"))


def ask_int(question: str, default: int, *, min_: int = 1) -> int:
    while True:
        try:
            raw = input(f"  {question} [{_c(str(default), 'yellow')}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise Aborted()
        _check_back(raw)
        if not raw:
            return default
        try:
            v = int(raw)
            if v < min_:
                raise ValueError
            return v
        except ValueError:
            print(_c(f"  please enter an integer >= {min_}", "magenta"))


# ---------------------------------------------------------------------------
# Profiles (preset defaults)
# ---------------------------------------------------------------------------


def _profile_defaults(profile: str) -> dict[str, Any]:
    base = {
        "embedder_model": "openai/text-embedding-3-small",
        "embedder_dim": 1536,
        "embedder_api_key_env": "OPENAI_API_KEY",
        "embedder_api_base": "",
        "llm_model": "openai/gpt-4o-mini",
        "llm_api_key_env": "OPENAI_API_KEY",
        "llm_api_base": "",
    }
    if profile == "dev":
        return {
            **base,
            "vector": "chromadb",
            "chroma_dir": "./storage/chroma",
            "blob": "local",
            "blob_root": "./storage/blobs",
        }
    if profile == "prod":
        return {
            **base,
            "pg_host": "localhost",
            "pg_port": 5432,
            "pg_database": "forgerag",
            "pg_user": "forgerag",
            "pg_password_env": "PG_PASSWORD",
            "vector": "pgvector",
            "embedder_dim": 1024,
            "blob": "local",
            "blob_root": "./storage/blobs",
        }
    return {}


# ---------------------------------------------------------------------------
# Connection tests
# ---------------------------------------------------------------------------


def _resolve_key(api_key: str | None, api_key_env: str | None) -> str | None:
    """Return the effective key: explicit > env. None if neither is set."""
    if api_key:
        return api_key
    if api_key_env:
        return os.environ.get(api_key_env)
    return None


def _test_embedding(model: str, key: str | None, base: str | None, *, timeout: float = 30.0) -> tuple[bool, str]:
    """Call litellm.embedding once with a short input. Returns (ok, message)."""
    try:
        from litellm import embedding  # type: ignore
    except ImportError as e:
        return False, f"litellm not installed: {e}"
    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "input": ["ping"],
            "timeout": timeout,
        }
        if key:
            kwargs["api_key"] = key
        if base:
            kwargs["api_base"] = base
        resp = embedding(**kwargs)
        # Probe the response shape so a misconfigured proxy that returns
        # 200 OK with non-embedding JSON still surfaces a clear error.
        data = getattr(resp, "data", None)
        if not data:
            return False, "response had no 'data' field"
        first = data[0]
        vec = first.get("embedding") if isinstance(first, dict) else getattr(first, "embedding", None)
        if not vec:
            return False, "first 'data' entry had no embedding"
        return True, f"ok (dim={len(vec)})"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _test_completion(model: str, key: str | None, base: str | None, *, timeout: float = 30.0) -> tuple[bool, str]:
    """Call litellm.completion once with a tiny prompt. Returns (ok, message)."""
    try:
        from litellm import completion  # type: ignore
    except ImportError as e:
        return False, f"litellm not installed: {e}"
    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
            "max_tokens": 10,
            "temperature": 0.0,
            "timeout": timeout,
        }
        if key:
            kwargs["api_key"] = key
        if base:
            kwargs["api_base"] = base
        resp = completion(**kwargs)
        choices = getattr(resp, "choices", None)
        if not choices:
            return False, "response had no choices"
        first = choices[0]
        msg = first.message if hasattr(first, "message") else first.get("message", {})
        text = (msg.content if hasattr(msg, "content") else msg.get("content", "")) or ""
        text = text.strip()[:60]
        return True, f"ok ({text!r})"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _confirm_test_failure(target: str, error: str) -> str:
    """
    Show the test error and ask what to do. Returns one of:
      "retry" — re-run the same step
      "back"  — go back to the previous step (raise _GoBack to caller)
      "skip"  — accept the values without a successful test
      "abort" — stop the wizard
    """
    print()
    print(_c(f"  ✗ {target} connection test FAILED:", "magenta"))
    print(_c(f"    {error}", "dim"))
    return ask_choice(
        "What now?",
        [
            ("retry", "fix the values and try again"),
            ("back",  "go back to the previous step"),
            ("skip",  "save anyway (you can fix it via /settings later)"),
            ("abort", "exit the wizard"),
        ],
        default="retry",
    )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _step_postgres(answers: dict, defaults: dict) -> None:
    answers["relational"] = "postgres"
    answers["pg_host"] = ask("Postgres host", default=defaults.get("pg_host", "localhost"))
    answers["pg_port"] = ask_int("Postgres port", default=defaults.get("pg_port", 5432))
    answers["pg_database"] = ask("Postgres database", default=defaults.get("pg_database", "forgerag"))
    answers["pg_user"] = ask("Postgres user", default=defaults.get("pg_user", "forgerag"))
    answers["pg_password_env"] = ask(
        "Env var containing the password",
        default=defaults.get("pg_password_env", "PG_PASSWORD"),
    )
    _ensure_backend_package(_RELATIONAL_PACKAGES, answers["relational"])


def _step_vector(answers: dict, defaults: dict) -> None:
    standalone = [
        ("chromadb", "ChromaDB — lightweight, backend-agnostic"),
        ("qdrant",   "Qdrant — production-grade, rich filtering"),
        ("milvus",   "Milvus — scalable, GPU-accelerated"),
        ("weaviate", "Weaviate — multi-modal, GraphQL API"),
    ]
    valid = [("pgvector", "pgvector — in-database, zero extra ops"), *standalone]
    default_vec = defaults.get("vector", valid[0][0])
    if default_vec not in [v for v, _ in valid]:
        default_vec = valid[0][0]
    answers["vector"] = ask_choice("Which vector backend?", valid, default=default_vec)
    if answers["vector"] == "chromadb":
        answers["chroma_dir"] = ask(
            "Chroma persist_directory",
            default=defaults.get("chroma_dir", "./storage/chroma"),
        )
    elif answers["vector"] == "qdrant":
        answers["qdrant_url"] = ask(
            "Qdrant server URL",
            default=defaults.get("qdrant_url", "http://localhost:6333"),
        )
    elif answers["vector"] == "milvus":
        answers["milvus_uri"] = ask(
            "Milvus server URI",
            default=defaults.get("milvus_uri", "http://localhost:19530"),
        )
    elif answers["vector"] == "weaviate":
        answers["weaviate_url"] = ask(
            "Weaviate server URL",
            default=defaults.get("weaviate_url", "http://localhost:8080"),
        )
    _ensure_backend_package(_VECTOR_PACKAGES, answers["vector"])


def _step_blob(answers: dict, defaults: dict) -> None:
    answers["blob"] = ask_choice(
        "Where should blobs live?",
        [
            ("local", "filesystem, single node"),
            ("s3",    "any S3-compatible service"),
            ("oss",   "Alibaba Cloud OSS"),
        ],
        default=defaults.get("blob", "local"),
    )
    if answers["blob"] == "local":
        answers["blob_root"] = ask(
            "Blob root directory",
            default=defaults.get("blob_root", "./storage/blobs"),
        )
    elif answers["blob"] == "s3":
        answers["s3_endpoint"] = ask("S3 endpoint URL", default="https://s3.amazonaws.com")
        answers["s3_bucket"] = ask("S3 bucket name")
        answers["s3_region"] = ask("S3 region", default="us-east-1")
        answers["s3_access_key_env"] = ask("Access key env var", default="S3_ACCESS_KEY")
        answers["s3_secret_key_env"] = ask("Secret key env var", default="S3_SECRET_KEY")
        answers["s3_public_base_url"] = ask(
            "Public CDN base URL (optional)",
            default="",
            allow_empty=True,
        )
    elif answers["blob"] == "oss":
        answers["oss_endpoint"] = ask(
            "OSS endpoint",
            default="https://oss-cn-hangzhou.aliyuncs.com",
        )
        answers["oss_bucket"] = ask("OSS bucket name")
        answers["oss_access_key_env"] = ask("Access key env var", default="OSS_ACCESS_KEY")
        answers["oss_secret_key_env"] = ask("Secret key env var", default="OSS_SECRET_KEY")
        answers["oss_public_base_url"] = ask(
            "Public base URL (optional)",
            default="",
            allow_empty=True,
        )
    _ensure_backend_package(_BLOB_PACKAGES, answers["blob"])


def _ask_credentials(prefix_label: str, defaults: dict, key_env: str, base_default: str) -> tuple[str, str, str]:
    """Common credential subform for embedder + LLM steps.

    Returns (api_key_env, api_key_plain, api_base). Exactly one of
    ``api_key_env`` and ``api_key_plain`` will be non-empty (the env var
    by default; falls through to plaintext only if the env var is unset
    and the operator pastes the key directly).
    """
    print(_c(f"  {prefix_label} authentication", "dim"))
    api_key_env = ask(
        "Env var containing the API key",
        default=defaults.get(key_env, "OPENAI_API_KEY"),
        allow_empty=True,
    )
    api_key_plain = ""
    if api_key_env and not os.environ.get(api_key_env):
        print(_c(f"  ! env var {api_key_env!r} is currently unset.", "yellow"))
        if ask_bool("Paste the key now (saved as plaintext in yaml)?", default=False):
            api_key_plain = ask("API key", allow_empty=False)
            api_key_env = ""  # plaintext takes precedence — clear the env reference
    api_base = ask(
        "Custom api_base (Ollama / OpenRouter / OneAPI / Azure URL)",
        default=defaults.get("llm_api_base" if "llm" in prefix_label.lower() else "embedder_api_base", base_default),
        allow_empty=True,
    )
    return api_key_env, api_key_plain, api_base


def _step_embedder(answers: dict, defaults: dict) -> None:
    while True:
        print(_c("  The embedding model converts text into vectors. Its output", "dim"))
        print(_c("  dimension MUST match your vector store's `dimension`.", "dim"))
        print(_c("  Common: 1536 (OpenAI small), 3072 (OpenAI large), 1024 (BGE-M3).", "dim"))
        answers["embedder_model"] = ask(
            "Embedding model (litellm format)",
            default=answers.get("embedder_model")
                or defaults.get("embedder_model", "openai/text-embedding-3-small"),
        )
        answers["embedder_dim"] = ask_int(
            "Embedding dimension",
            default=answers.get("embedder_dim") or defaults.get("embedder_dim", 1536),
        )
        api_key_env, api_key_plain, api_base = _ask_credentials(
            "Embedder", defaults, "embedder_api_key_env", ""
        )
        answers["embedder_api_key_env"] = api_key_env
        answers["embedder_api_key"] = api_key_plain
        answers["embedder_api_base"] = api_base

        # Live test
        print()
        print(_c("  testing embedding endpoint…", "dim"))
        key = _resolve_key(api_key_plain, api_key_env)
        ok, msg = _test_embedding(answers["embedder_model"], key, api_base or None)
        if ok:
            print(_c(f"  ✓ {msg}", "green"))
            return
        choice = _confirm_test_failure("Embedding", msg)
        if choice == "retry":
            continue
        if choice == "back":
            raise _GoBack()
        if choice == "abort":
            raise Aborted()
        # "skip" — accept and move on
        return


def _step_llm(answers: dict, defaults: dict) -> None:
    while True:
        print(_c("  The answer-generation LLM produces the final answer text.", "dim"))
        print(_c("  Any litellm-compatible model works (OpenAI / Anthropic / DeepSeek /", "dim"))
        print(_c("  Ollama / OpenRouter / Azure / Bedrock / Vertex / ...).", "dim"))
        answers["llm_model"] = ask(
            "Generator model (litellm format)",
            default=answers.get("llm_model")
                or defaults.get("llm_model", "openai/gpt-4o-mini"),
        )
        api_key_env, api_key_plain, api_base = _ask_credentials(
            "LLM", defaults, "llm_api_key_env", ""
        )
        answers["llm_api_key_env"] = api_key_env
        answers["llm_api_key"] = api_key_plain
        answers["llm_api_base"] = api_base

        # Live test
        print()
        print(_c("  testing generator endpoint (one short completion call)…", "dim"))
        key = _resolve_key(api_key_plain, api_key_env)
        ok, msg = _test_completion(answers["llm_model"], key, api_base or None)
        if ok:
            print(_c(f"  ✓ {msg}", "green"))
            return
        choice = _confirm_test_failure("LLM", msg)
        if choice == "retry":
            continue
        if choice == "back":
            raise _GoBack()
        if choice == "abort":
            raise Aborted()
        # "skip" — accept and move on
        return


# ---------------------------------------------------------------------------
# The wizard
# ---------------------------------------------------------------------------


_STEPS: list[tuple[str, Callable[[dict, dict], None]]] = [
    ("Metadata database (PostgreSQL)", _step_postgres),
    ("Vector database",                _step_vector),
    ("Blob storage",                   _step_blob),
    ("Embedding model",                _step_embedder),
    ("Answer-generation LLM",          _step_llm),
]


def _non_interactive_defaults(profile: str) -> dict[str, Any]:
    d = _profile_defaults(profile)
    if not d:
        return d
    # Fill in fields the per-step functions would normally set so
    # build_config_dict has everything it needs without prompting.
    d.setdefault("relational", "postgres")
    d.setdefault("pg_host", "localhost")
    d.setdefault("pg_port", 5432)
    d.setdefault("pg_database", "forgerag")
    d.setdefault("pg_user", "forgerag")
    d.setdefault("pg_password_env", "PG_PASSWORD")
    d.setdefault("blob_root", "./storage/blobs")
    if d.get("vector") == "chromadb":
        d.setdefault("chroma_dir", "./storage/chroma")
    d.setdefault("embedder_api_key", "")
    d.setdefault("llm_api_key", "")
    return d


def run_wizard(profile: str, non_interactive: bool) -> dict[str, Any]:
    """Return a dict of answers that the yaml builder consumes."""
    defaults = _profile_defaults(profile)

    if non_interactive:
        d = _non_interactive_defaults(profile)
        if not d:
            print("error: --non-interactive requires --profile dev|prod", file=sys.stderr)
            raise Aborted()
        return d

    banner("ForgeRAG setup wizard")
    print(_c("  Press Enter to accept the default in [yellow].", "dim"))
    print(_c("  Type 'b' / 'back' / '<' to re-open the previous step.", "dim"))
    print(_c("  Ctrl-C to abort.", "dim"))

    answers: dict[str, Any] = {}
    i = 0
    while i < len(_STEPS):
        title, fn = _STEPS[i]
        section(f"{i + 1}/{len(_STEPS)}  {title}")
        try:
            fn(answers, defaults)
        except _GoBack:
            if i == 0:
                print(_c("  already at the first step — nowhere to go back.", "yellow"))
                continue
            i -= 1
            continue
        i += 1

    section("Done!")
    return answers


# ---------------------------------------------------------------------------
# YAML builder
# ---------------------------------------------------------------------------


def build_config_dict(a: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}

    # --- parser (minimal: pymupdf always on; MinerU via /settings) ---
    cfg["parser"] = {"backends": {"pymupdf": {"enabled": True}}}

    # --- storage (blob) ---
    storage: dict[str, Any] = {"mode": a["blob"]}
    if a["blob"] == "local":
        storage["local"] = {"root": a["blob_root"]}
    elif a["blob"] == "s3":
        storage["s3"] = {
            "endpoint": a["s3_endpoint"],
            "bucket": a["s3_bucket"],
            "region": a["s3_region"],
            "access_key_env": a["s3_access_key_env"],
            "secret_key_env": a["s3_secret_key_env"],
        }
        if a.get("s3_public_base_url"):
            storage["s3"]["public_base_url"] = a["s3_public_base_url"]
    elif a["blob"] == "oss":
        storage["oss"] = {
            "endpoint": a["oss_endpoint"],
            "bucket": a["oss_bucket"],
            "access_key_env": a["oss_access_key_env"],
            "secret_key_env": a["oss_secret_key_env"],
        }
        if a.get("oss_public_base_url"):
            storage["oss"]["public_base_url"] = a["oss_public_base_url"]
    cfg["storage"] = storage

    # --- persistence ---
    rel: dict[str, Any] = {"backend": "postgres"}
    rel["postgres"] = {
        "host": a["pg_host"],
        "port": a["pg_port"],
        "database": a["pg_database"],
        "user": a["pg_user"],
        "password_env": a["pg_password_env"],
    }

    vec: dict[str, Any] = {"backend": a["vector"]}
    if a["vector"] == "pgvector":
        vec["pgvector"] = {
            "dimension": a["embedder_dim"],
            "index_type": "hnsw",
            "distance": "cosine",
        }
    elif a["vector"] == "chromadb":
        vec["chromadb"] = {
            "mode": "persistent",
            "persist_directory": a["chroma_dir"],
            "collection_name": "forgerag",
            "dimension": a["embedder_dim"],
            "distance": "cosine",
        }
    elif a["vector"] == "qdrant":
        vec["qdrant"] = {
            "url": a["qdrant_url"],
            "collection_name": "forgerag_chunks",
            "dimension": a["embedder_dim"],
            "distance": "cosine",
        }
    elif a["vector"] == "milvus":
        vec["milvus"] = {
            "uri": a["milvus_uri"],
            "collection_name": "forgerag_chunks",
            "dimension": a["embedder_dim"],
            "distance": "cosine",
            "index_type": "HNSW",
        }
    elif a["vector"] == "weaviate":
        vec["weaviate"] = {
            "url": a["weaviate_url"],
            "collection_name": "ForgeragChunks",
            "dimension": a["embedder_dim"],
            "distance": "cosine",
        }

    cfg["persistence"] = {"relational": rel, "vector": vec}

    # --- embedder ---
    embedder_litellm: dict[str, Any] = {"model": a["embedder_model"]}
    if a.get("embedder_api_key"):
        embedder_litellm["api_key"] = a["embedder_api_key"]
    elif a.get("embedder_api_key_env"):
        embedder_litellm["api_key_env"] = a["embedder_api_key_env"]
    if a.get("embedder_api_base"):
        embedder_litellm["api_base"] = a["embedder_api_base"]
    cfg["embedder"] = {
        "backend": "litellm",
        "dimension": a["embedder_dim"],
        "litellm": embedder_litellm,
    }

    # --- answering generator ---
    generator: dict[str, Any] = {"backend": "litellm", "model": a["llm_model"]}
    if a.get("llm_api_key"):
        generator["api_key"] = a["llm_api_key"]
    elif a.get("llm_api_key_env"):
        generator["api_key_env"] = a["llm_api_key_env"]
    if a.get("llm_api_base"):
        generator["api_base"] = a["llm_api_base"]
    cfg["answering"] = {"generator": generator}

    # --- files ---
    cfg["files"] = {"hash_algorithm": "sha256", "max_bytes": 524288000}

    return cfg


def write_yaml(cfg: dict[str, Any], path: Path) -> None:
    try:
        import yaml
    except ImportError:
        raise RuntimeError("pyyaml not installed: pip install pyyaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )


# ---------------------------------------------------------------------------
# Post-setup actions
# ---------------------------------------------------------------------------


def _child_env() -> dict[str, str]:
    """Subprocess env that inherits ours plus forces UTF-8 stdio so child
    Python processes don't crash on box-drawing / arrow chars under
    Windows GBK consoles."""
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def post_setup(config_path: Path) -> None:
    section("Next steps")

    # Run config validator (subprocess isolates us from pydantic issues)
    try:
        r = subprocess.run(
            [sys.executable, "-m", "config", "validate", str(config_path)],
            cwd=_ROOT,
            env=_child_env(),
        )
        if r.returncode != 0:
            print(_c("  config validation FAILED — fix the file and re-run", "magenta"))
            return
    except FileNotFoundError:
        pass

    print()
    choice = ask_choice(
        "What do you want to do next?",
        [
            ("nothing", "just exit; run it yourself later"),
            ("batch",   "batch-ingest files from a directory now"),
            ("api",     "start the HTTP API (uvicorn) now"),
        ],
        default="nothing",
    )
    if choice == "nothing":
        print()
        print(_c("  done. to use this config later:", "dim"))
        print(f"    export FORGERAG_CONFIG={config_path}")
        return

    if choice == "batch":
        target = ask(
            "Directory to ingest",
            default="./papers",
            validator=lambda p: None if Path(p).exists() else f"not found: {p}",
        )
        embed = ask_bool("Compute embeddings?", default=False)
        cmd = [
            sys.executable,
            "scripts/batch_ingest.py",
            target,
            "--config",
            str(config_path),
        ]
        if embed:
            cmd.append("--embed")
        print(_c(f"\n  running: {' '.join(cmd)}\n", "dim"))
        subprocess.run(cmd, cwd=_ROOT, env=_child_env())
        return

    if choice == "api":
        host = ask("Host", default="0.0.0.0")
        port = ask_int("Port", default=8000)
        env = _child_env()
        env["FORGERAG_CONFIG"] = str(config_path)
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "api.app:app",
            "--host",
            host,
            "--port",
            str(port),
            "--reload",
        ]
        print(_c(f"\n  FORGERAG_CONFIG={config_path}", "dim"))
        print(_c(f"  running: {' '.join(cmd)}\n", "dim"))
        try:
            subprocess.run(cmd, cwd=_ROOT, env=env)
        except KeyboardInterrupt:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_HELP_DESCRIPTION = """\
Interactive setup wizard for ForgeRAG.

Walks through five small steps and writes a forgerag.yaml that wires
together the relational store, vector store, blob storage, embedder,
and answer-generation LLM. The embedder and LLM steps each finish with
a real connection test (live API call) so a typo in api_base or a
wrong key surfaces immediately.

Type 'b' / 'back' / '<' at any prompt to re-open the previous step.

All LLM / embedding backends go through litellm, which accepts custom
endpoints via an api_base setting -- so Ollama, vLLM, OneAPI,
OpenRouter, DeepSeek, Azure, any OpenAI-compatible server all work
with the same model string.
"""

_HELP_EPILOG = """\
Profiles
--------
  dev    ChromaDB + local blob + OpenAI defaults.
         Zero infrastructure beyond Postgres, good for local exper.

  prod   pgvector + local blob + OpenAI defaults.
         Recommended for a single production node.

  custom Full wizard, no presets. Use when you know what you want.

LLM / model configuration
-------------------------
  Model + api_key + api_base ARE set here so the wizard can verify
  the connection live. Other tunables (temperature, prompts,
  retrieval strategy) remain runtime-editable via /settings.

Typical runs
------------
  # Interactive wizard with prod preset, save to myconfig.yaml:
  python scripts/setup.py --profile prod -o myconfig.yaml

  # Non-interactive dev (zero prompts, for CI / Docker):
  python scripts/setup.py --profile dev -y

  # Full custom wizard:
  python scripts/setup.py --profile custom

Using the generated config afterwards
-------------------------------------
  # Validate:
  python -m config validate forgerag.yaml

  # Point everything at it:
  export FORGERAG_CONFIG=./forgerag.yaml

  # Batch-ingest some files:
  python scripts/batch_ingest.py ./papers

  # Launch the HTTP API:
  uvicorn api.app:app --host 0.0.0.0 --port 8000

"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="setup.py",
        description=_HELP_DESCRIPTION,
        epilog=_HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--profile",
        choices=("dev", "prod", "custom"),
        default="custom",
        help="Preset defaults. dev=chromadb+local; prod=pgvector+local.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./forgerag.yaml"),
        help="Where to write the generated config.",
    )
    p.add_argument(
        "-y",
        "--non-interactive",
        action="store_true",
        help="Accept profile defaults without prompting. Requires --profile.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.output.exists() and not args.force:
        print(_c(f"  {args.output} already exists. Use --force to overwrite.", "magenta"))
        return 2

    try:
        answers = run_wizard(args.profile, args.non_interactive)
    except Aborted:
        print("\n  aborted.")
        return 130

    cfg_dict = build_config_dict(answers)
    try:
        write_yaml(cfg_dict, args.output)
    except Exception as e:
        print(_c(f"  failed to write {args.output}: {e}", "magenta"))
        return 1

    print()
    print(_c(f"  wrote {args.output}", "green"))
    print()

    try:
        post_setup(args.output)
    except Aborted:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
