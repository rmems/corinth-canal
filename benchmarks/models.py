"""Inference backends used by the Python benchmark harness.

All third-party runtimes are optional and imported only when an adapter is
loaded.  Importing this module is therefore safe in the CPU-only CI profile.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import shutil
import subprocess  # nosec B404
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# llama-cli default subprocess timeout when no generation/manifest limit is set.
# Cold GGUF load plus a short completion commonly exceeds 30s; 600s is generous
# without hanging the harness indefinitely.
_DEFAULT_LLAMA_CLI_TIMEOUT_S = 600.0
_LLAMA_CLI_NAME = "llama-cli"
_MODEL_PATH_FIELDS = (
    "path",
    "model_path",
    "model_id_or_path",
    "model_id_or_local_path",
    "checkpoint_path",
    "artifact_path",
    "artifact",
    "model_id",
)


class BackendUnavailableError(RuntimeError):
    """Raised when a selected optional inference runtime is unavailable."""


def _field(manifest: Mapping[str, Any] | object, name: str, default: Any = None) -> Any:
    if isinstance(manifest, Mapping):
        return manifest.get(name, default)
    return getattr(manifest, name, default)


def _model_path(manifest: Mapping[str, Any] | object) -> str:
    for field in _MODEL_PATH_FIELDS:
        value = _field(manifest, field)
        if value:
            return str(value)
    raise ValueError(
        "model manifest must define path, model_path, model_id_or_path, "
        "model_id_or_local_path, checkpoint_path, artifact_path, artifact, or model_id"
    )


def _has_model_path(manifest: Mapping[str, Any] | object) -> bool:
    try:
        _model_path(manifest)
    except ValueError:
        return False
    return True


def _require_integral_tokens(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"max_tokens must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"max_tokens must be an integer, got {value!r}")
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        sign = stripped[:1] in {"+", "-"}
        digits = stripped[1:] if sign else stripped
        if digits.isdigit():
            return int(stripped)
    raise ValueError(f"max_tokens must be an integer, got {value!r}")


def _require_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric, got {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _validate_generation(generation: dict[str, Any]) -> tuple[int, float, float]:
    """Validate and normalize generation options."""
    max_tokens = _require_integral_tokens(generation.get("max_tokens", 32))
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
    if max_tokens > 4096:
        raise ValueError(f"max_tokens {max_tokens} exceeds limit 4096")
    temperature = _require_finite_float(generation.get("temperature", 0.0), "temperature")
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    top_p = _require_finite_float(generation.get("top_p", 1.0), "top_p")
    if not 0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    return max_tokens, temperature, top_p


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _resolve_llama_cli(configured: str | None) -> str:
    """Return a validated absolute llama-cli path.

    An explicit ``executable`` may be any absolute existing file with the
    execute bit. PATH lookup only accepts a binary named ``llama-cli``.
    """
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise BackendUnavailableError(
                f"llama.cpp executable must be an absolute path, got {configured!r}"
            )
        if not _is_executable_file(path):
            raise BackendUnavailableError(
                f"llama.cpp executable is missing or not executable: {configured!r}"
            )
        return str(path)

    found = shutil.which(_LLAMA_CLI_NAME)
    if not found:
        raise BackendUnavailableError(
            "llama.cpp backend is unavailable: install llama-cpp-python or put "
            "llama-cli on PATH (an executable may also be passed explicitly)"
        )
    path = Path(found)
    if path.name != _LLAMA_CLI_NAME or not path.is_absolute() or not _is_executable_file(path):
        raise BackendUnavailableError(
            f"refusing to run {found!r}; expected an executable named {_LLAMA_CLI_NAME}"
        )
    return str(path)


def _invoke_llama_cli(
    executable: str,
    model_path: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
) -> str:
    """Run one validated llama-cli turn. argv[0] is the literal name ``llama-cli``;
    ``executable`` is the already-checked absolute path.
    """
    try:
        # executable was validated by _resolve_llama_cli (absolute path + execute bit).
        completed = subprocess.run(  # nosec S603,B603
            [
                "llama-cli",
                "-m",
                model_path,
                "-p",
                prompt,
                "-n",
                str(max_tokens),
                "--temp",
                str(temperature),
                "--top-p",
                str(top_p),
                "--no-display-prompt",
                "-no-cnv",
            ],
            executable=executable,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            shell=False,  # nosec S603,B603
        )
    except PermissionError as error:
        raise BackendUnavailableError(
            f"llama.cpp executable is not executable: {executable!r}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"llama-cli timed out after {timeout}s: {error}") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"llama-cli failed: {error.stderr}") from error
    text = completed.stdout.strip()
    if not text:
        raise RuntimeError("llama-cli returned empty output")
    return text


def _llama_cli_timeout(
    generation: Mapping[str, Any],
    options: Mapping[str, Any],
    manifest: Mapping[str, Any] | object,
) -> float:
    """Subprocess timeout in seconds for a single llama-cli turn.

    Preference: generation ``timeout``, then adapter option ``timeout``, then
    manifest ``max_runtime_minutes`` (converted to seconds). Otherwise
    :data:`_DEFAULT_LLAMA_CLI_TIMEOUT_S` (600s).
    """
    if "timeout" in generation:
        raw = generation["timeout"]
        name = "timeout"
    elif "timeout" in options:
        raw = options["timeout"]
        name = "timeout"
    else:
        raw = _field(manifest, "max_runtime_minutes")
        if raw in (None, ""):
            return _DEFAULT_LLAMA_CLI_TIMEOUT_S
        name = "max_runtime_minutes"
        timeout = _require_finite_float(raw, name) * 60.0
        if timeout <= 0:
            raise ValueError(f"{name} must be > 0, got {raw!r}")
        return timeout
    timeout = _require_finite_float(raw, name)
    if timeout <= 0:
        raise ValueError(f"{name} must be > 0, got {raw!r}")
    return timeout


class ModelAdapter(ABC):
    """Small common interface implemented by every benchmark backend."""

    backend: str

    def __init__(self, manifest: Mapping[str, Any] | object, **options: Any) -> None:
        self.manifest = manifest
        self.model_path = _model_path(manifest)
        self.options = options
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load the model artifact and its runtime."""

    @abstractmethod
    def generate(self, prompt: str, **generation: Any) -> str:
        """Generate text for one prompt."""

    def predict(self, prompt: str, **generation: Any) -> str:
        """Compatibility alias used by benchmark evaluators."""
        return self.generate(prompt, **generation)


class MockModelAdapter(ModelAdapter):
    """Dependency-free adapter retained for smoke tests and development."""

    backend = "mock"

    def __init__(self, manifest: Mapping[str, Any] | object | None = None, **options: Any) -> None:
        if manifest is None or not _has_model_path(manifest):
            base = dict(manifest) if isinstance(manifest, Mapping) else {}
            base.setdefault("path", "mock")
            manifest = base
        super().__init__(manifest, **options)

    def load(self) -> None:
        self._loaded = True

    def generate(self, prompt: str, **generation: Any) -> str:
        if not self._loaded:
            self.load()
        return str(generation.get("response", self.options.get("response", prompt)))


class LlamaCppAdapter(ModelAdapter):
    """GGUF inference through llama-cpp-python or the llama CLI."""

    backend = "llama.cpp"

    def __init__(self, manifest: Mapping[str, Any] | object, **options: Any) -> None:
        super().__init__(manifest, **options)
        self._mode: str | None = None
        self._executable: str | None = None
        self._model: Any | None = None

    def load(self) -> None:
        if self._loaded:
            return
        try:
            module = importlib.import_module("llama_cpp")
        except ImportError:
            module = None
        if module is not None:
            kwargs = {
                "n_ctx": self.options.get("n_ctx", 2048),
                "n_gpu_layers": self.options.get("n_gpu_layers", 0),
                "verbose": self.options.get("verbose", False),
            }
            self._model = module.Llama(model_path=self.model_path, **kwargs)
            self._mode = "python"
            self._loaded = True
            return

        configured = self.options.get("executable")
        self._executable = _resolve_llama_cli(str(configured) if configured else None)
        self._mode = "cli"
        self._loaded = True

    def generate(self, prompt: str, **generation: Any) -> str:
        if not getattr(self, "_loaded", False):
            try:
                self.load()
            except BackendUnavailableError:
                raise
            except Exception as error:
                raise BackendUnavailableError(
                    f"llama.cpp backend is unavailable: {error}"
                ) from error
        if not getattr(self, "_loaded", False) or getattr(self, "_mode", None) is None:
            raise BackendUnavailableError(
                "llama.cpp backend is unavailable: adapter did not finish loading"
            )
        max_tokens, temperature, top_p = _validate_generation(generation)
        if self._mode == "python":
            if self._model is None:
                raise BackendUnavailableError(
                    "llama.cpp python backend reported loaded but holds no model"
                )
            result = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False,
            )
            return str(result["choices"][0]["text"])
        if self._executable is None:
            raise BackendUnavailableError(
                "llama.cpp cli backend reported loaded but holds no executable"
            )
        timeout = _llama_cli_timeout(generation, self.options, self.manifest)
        return _invoke_llama_cli(
            self._executable,
            self.model_path,
            prompt,
            max_tokens,
            temperature,
            top_p,
            timeout,
        )


class VllmAdapter(ModelAdapter):
    """High-throughput inference for Hugging Face/Safetensors artifacts."""

    backend = "vllm"

    def load(self) -> None:
        if self._loaded:
            return
        try:
            module = importlib.import_module("vllm")
        except ImportError as error:
            raise BackendUnavailableError(
                "vLLM backend is unavailable: install the optional 'vllm' package"
            ) from error
        # Filter options to known safe keys; do not blanket-forward trust_remote_code
        allowed_keys = {
            "dtype",
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "max_model_len",
            "enforce_eager",
        }
        kwargs = {k: v for k, v in self.options.items() if k in allowed_keys}
        if self.options.get("trust_remote_code"):
            warnings.warn(
                "trust_remote_code=True is not supported for vLLM adapter; ignoring",
                stacklevel=2,
            )
        self._sampling_params = module.SamplingParams
        self._model = module.LLM(model=self.model_path, **kwargs)
        self._loaded = True

    def generate(self, prompt: str, **generation: Any) -> str:
        if not self._loaded:
            self.load()
        max_tokens, temperature, top_p = _validate_generation(generation)
        params = self._sampling_params(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        result = self._model.generate([prompt], params, use_tqdm=False)
        return str(result[0].outputs[0].text)


def _hf_trust_remote_code(options: Mapping[str, Any]) -> bool:
    trust = options.get("trust_remote_code", False)
    if not isinstance(trust, bool):
        raise ValueError("trust_remote_code must be a boolean")
    if trust:
        warnings.warn(
            "trust_remote_code=True allows arbitrary code execution from model files; "
            "only enable for trusted models",
            UserWarning,
            stacklevel=3,
        )
    return trust


def _hf_from_pretrained_kwargs(options: Mapping[str, Any]) -> dict[str, Any]:
    model_options = {
        key: value
        for key, value in options.items()
        if key not in {"trust_remote_code", "executable"}
    }
    allowed = {"torch_dtype", "device_map", "low_cpu_mem_usage", "attn_implementation"}
    filtered = {key: value for key, value in model_options.items() if key in allowed}
    if len(filtered) != len(model_options):
        warnings.warn(
            f"Ignoring unsupported HF options: {set(model_options) - set(filtered)}",
            stacklevel=3,
        )
    return filtered


def _first_param_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


class HuggingFaceAdapter(ModelAdapter):
    """Transformers fallback for standard causal language models."""

    backend = "huggingface"

    def load(self) -> None:
        if self._loaded:
            return
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise BackendUnavailableError(
                "Hugging Face backend is unavailable: install the optional "
                "'transformers' package (and a supported tensor runtime)"
            ) from error
        trust = _hf_trust_remote_code(self.options)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=trust
        )
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=trust,
            **_hf_from_pretrained_kwargs(self.options),
        )
        self._device = _first_param_device(self._model)
        self._loaded = True

    def generate(self, prompt: str, **generation: Any) -> str:
        if not self._loaded:
            self.load()
        max_tokens, temperature, top_p = _validate_generation(generation)
        encoded = self._tokenizer(prompt, return_tensors="pt")
        device = getattr(self, "_device", None)
        if device is not None:
            if hasattr(encoded, "to"):
                encoded = encoded.to(device)
            else:
                encoded = {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in encoded.items()
                }
        do_sample = temperature > 0
        output = self._model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p,
        )
        prompt_tokens = encoded["input_ids"].shape[-1]
        return self._tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True)


_GGUF_FORMATS = {
    "gguf",
    "q4_k_m",
    "q4_0",
    "q4_k_s",
    "q5_k_m",
    "q5_0",
    "q6_k",
    "q8_0",
    "iq4_nl",
    "iq3_m",
    "q2_k",
    "q3_k_m",
}
_HF_FORMATS = {
    "safetensors",
    "hf",
    "huggingface",
    "bf16",
    "fp16",
    "fp8",
    "int8",
    "bnb-4bit",
    "bnb_4bit",
    "gptq",
    "awq",
    "pt",
    "bin",
}
_EXPLICIT_RUNTIMES = {
    "llama.cpp": "llama.cpp",
    "llama_cpp": "llama.cpp",
    "llamacpp": "llama.cpp",
    "gguf": "llama.cpp",
    "vllm": "vllm",
    "hf": "huggingface",
    "huggingface": "huggingface",
    "hf_transformers": "huggingface",
    "transformers": "huggingface",
    "mock": "mock",
}


def _backend_from_runtime(runtime: str) -> str | None:
    return _EXPLICIT_RUNTIMES.get(runtime)


def _resolved_model_path_lower(manifest: Mapping[str, Any] | object) -> str:
    try:
        return _model_path(manifest).lower()
    except ValueError:
        return ""


def _backend_from_artifact(source: str, model_path_str: str) -> str | None:
    if source in _GGUF_FORMATS or model_path_str.endswith(".gguf"):
        return "llama.cpp"
    if (
        source in _HF_FORMATS
        or source.startswith("safetensors")
        or model_path_str.endswith(".safetensors")
    ):
        return "vllm"
    return None


# Caller-supplied override. No producer in this repository writes it, but it
# lets a runner force a backend without touching artifact metadata.
_RUNTIME_OVERRIDE_FIELD = "runtime_format"

# Metadata fields that describe the artifact, most specific first:
#
#   loader_hint        ModelAdapterConfig  (configs/model_adapter_configs.toml)
#   checkpoint_format  ExperimentManifest  (run_manifest.json)
#   source_format      RunMatrix entries
#   format             ModelAdapterConfig
#
# The *value* is authoritative, not the field name: `loader_hint` is documented
# as a loader hint but every entry on disk carries "safetensors" or "gguf",
# which are artifact formats. So each value is tried as a runtime name first and
# as an artifact format second.
_BACKEND_HINT_FIELDS = (
    "loader_hint",
    "checkpoint_format",
    "source_format",
    "format",
)


def _normalized_field(manifest: Mapping[str, Any] | object, name: str) -> str:
    value = _field(manifest, name, "")
    if not value:
        return ""
    return str(value).strip().lower().replace("-", "_")


def detect_backend(manifest: Mapping[str, Any] | object) -> str:
    """Choose a backend from explicit runtime and artifact format metadata.

    Accepts the field vocabulary the Rust producers actually emit
    (`checkpoint_format`, `source_format`, `format`, `loader_hint`) rather than
    a single key nothing writes.
    """
    override = _normalized_field(manifest, _RUNTIME_OVERRIDE_FIELD)
    if override:
        backend = _backend_from_runtime(override)
        if backend is None:
            raise ValueError(
                f"unsupported {_RUNTIME_OVERRIDE_FIELD} {override!r}; expected one of "
                f"{sorted(set(_EXPLICIT_RUNTIMES))}"
            )
        return backend

    for field in _BACKEND_HINT_FIELDS:
        value = _normalized_field(manifest, field)
        if not value:
            continue
        if value == "custom_artifact":
            raise ValueError(
                f"{field} 'custom_artifact' names no inference runtime; set "
                f"{_RUNTIME_OVERRIDE_FIELD} explicitly for this model"
            )
        backend = _backend_from_runtime(value) or _backend_from_artifact(value, "")
        if backend is not None:
            return backend

    backend = _backend_from_artifact("", _resolved_model_path_lower(manifest))
    if backend is None:
        raise ValueError(
            "cannot detect inference backend; set "
            f"{_RUNTIME_OVERRIDE_FIELD} or one of {list(_BACKEND_HINT_FIELDS)} "
            "(saw none, and the model path has no recognised suffix)"
        )
    return backend


def adapter_for_manifest(
    manifest: Mapping[str, Any] | object, *, load: bool = False, **options: Any
) -> ModelAdapter:
    """Construct (and optionally load) the adapter selected by a manifest."""
    adapters = {
        "llama.cpp": LlamaCppAdapter,
        "vllm": VllmAdapter,
        "huggingface": HuggingFaceAdapter,
        "mock": MockModelAdapter,
    }
    adapter = adapters[detect_backend(manifest)](manifest, **options)
    if load:
        adapter.load()
    return adapter


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a JSON model manifest for simple runner integrations."""
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("model manifest must be a JSON object")
    return value
