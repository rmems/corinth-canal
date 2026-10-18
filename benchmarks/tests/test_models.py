from __future__ import annotations

import math
import subprocess  # nosec B404
import sys
import tempfile
import tomllib
import types
import unittest
from pathlib import Path
from unittest import mock

from benchmarks import load_manifest as exported_load_manifest
from benchmarks.models import (
    BackendUnavailableError,
    HuggingFaceAdapter,
    LlamaCppAdapter,
    MockModelAdapter,
    VllmAdapter,
    _hf_trust_remote_code,
    _validate_generation,
    adapter_for_manifest,
    detect_backend,
    load_manifest,
)


class BackendDetectionTests(unittest.TestCase):
    def test_detects_gguf_and_safetensors(self) -> None:
        self.assertEqual(
            detect_backend({"path": "tiny.gguf", "source_format": "GGUF"}), "llama.cpp"
        )
        self.assertEqual(detect_backend({"path": "org/model", "source_format": "BF16"}), "vllm")

    def test_detects_suffixes_on_path_aliases(self) -> None:
        self.assertEqual(detect_backend({"model_path": "tiny.gguf"}), "llama.cpp")
        self.assertEqual(detect_backend({"artifact_path": "model.safetensors"}), "vllm")
        self.assertEqual(detect_backend({"model_id_or_path": "weights.gguf"}), "llama.cpp")
        self.assertEqual(detect_backend({"model_id_or_local_path": "shard.safetensors"}), "vllm")
        self.assertEqual(detect_backend({"checkpoint_path": "ckpt.safetensors"}), "vllm")

    def test_explicit_runtime_wins(self) -> None:
        manifest = {
            "path": "org/model",
            "source_format": "safetensors",
            "runtime_format": "transformers",
        }
        self.assertIsInstance(adapter_for_manifest(manifest), HuggingFaceAdapter)

    def test_unknown_metadata_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot detect inference backend"):
            detect_backend({"path": "model.bin"})

    def test_pathless_explicit_mock_manifest(self) -> None:
        adapter = adapter_for_manifest({"runtime_format": "mock"})
        self.assertIsInstance(adapter, MockModelAdapter)
        self.assertEqual(adapter.generate("hello"), "hello")


class PathAliasTests(unittest.TestCase):
    def test_adapter_accepts_canonical_path_fields(self) -> None:
        cases = {
            "model_id_or_path": "from-run-entry.gguf",
            "model_id_or_local_path": "from-adapter-config.gguf",
            "checkpoint_path": "from-experiment.gguf",
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                adapter = LlamaCppAdapter({field: value})
                self.assertEqual(adapter.model_path, value)


class GenerationValidationTests(unittest.TestCase):
    def test_rejects_fractional_and_boolean_max_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            _validate_generation({"max_tokens": 31.9})
        with self.assertRaisesRegex(ValueError, "integer"):
            _validate_generation({"max_tokens": True})

    def test_accepts_integer_max_tokens(self) -> None:
        max_tokens, _, _ = _validate_generation({"max_tokens": 64})
        self.assertEqual(max_tokens, 64)
        max_tokens, _, _ = _validate_generation({"max_tokens": "16"})
        self.assertEqual(max_tokens, 16)

    def test_rejects_non_finite_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            _validate_generation({"temperature": math.nan})
        with self.assertRaisesRegex(ValueError, "finite"):
            _validate_generation({"temperature": math.inf})
        with self.assertRaisesRegex(ValueError, "finite"):
            _validate_generation({"temperature": -math.inf})


class TrustRemoteCodeTests(unittest.TestCase):
    def test_defaults_false_and_accepts_bool(self) -> None:
        self.assertFalse(_hf_trust_remote_code({}))
        self.assertFalse(_hf_trust_remote_code({"trust_remote_code": False}))
        self.assertTrue(_hf_trust_remote_code({"trust_remote_code": True}))

    def test_rejects_non_boolean_values(self) -> None:
        for value in ("false", "true", 1, 0, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "boolean"):
                    _hf_trust_remote_code({"trust_remote_code": value})


class OptionalBackendTests(unittest.TestCase):
    def test_llama_python_backend_generates_text(self) -> None:
        class FakeLlama:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": " real answer"}]}

        module = types.SimpleNamespace(Llama=FakeLlama)
        with mock.patch.dict(sys.modules, {"llama_cpp": module}):
            adapter = LlamaCppAdapter({"path": "tiny.gguf"})
            self.assertEqual(adapter.generate("question"), " real answer")

    def test_llama_python_forwards_top_p(self) -> None:
        calls: list[dict] = []

        class FakeLlama:
            def __init__(self, **kwargs):
                pass

            def __call__(self, prompt, **kwargs):
                calls.append(kwargs)
                return {"choices": [{"text": "sampled"}]}

        module = types.SimpleNamespace(Llama=FakeLlama)
        with mock.patch.dict(sys.modules, {"llama_cpp": module}):
            adapter = LlamaCppAdapter({"path": "tiny.gguf"})
            self.assertEqual(adapter.generate("question", top_p=0.7, temperature=0.8), "sampled")
        self.assertEqual(calls[0]["top_p"], 0.7)
        self.assertEqual(calls[0]["temperature"], 0.8)

    def test_llama_cli_forwards_top_p_and_honors_timeout(self) -> None:
        adapter = LlamaCppAdapter({"path": "tiny.gguf"})
        adapter._loaded = True
        adapter._mode = "cli"
        adapter._executable = "/opt/llama-cli"
        completed = mock.Mock(stdout="cli answer\n")
        with mock.patch("benchmarks.models.subprocess.run", return_value=completed) as run:
            text = adapter.generate("prompt", top_p=0.55, timeout=12)
        self.assertEqual(text, "cli answer")
        command = run.call_args.args[0]
        self.assertEqual(command[0], "llama-cli")
        self.assertEqual(run.call_args.kwargs["executable"], "/opt/llama-cli")
        self.assertIn("--top-p", command)
        self.assertEqual(command[command.index("--top-p") + 1], "0.55")
        self.assertIn("-no-cnv", command)
        self.assertEqual(run.call_args.kwargs["timeout"], 12)
        self.assertIsNotNone(run.call_args.kwargs.get("stdin"))
        self.assertFalse(run.call_args.kwargs["shell"])

    def test_llama_cli_empty_stdout_is_not_success(self) -> None:
        adapter = LlamaCppAdapter({"path": "tiny.gguf"})
        adapter._loaded = True
        adapter._mode = "cli"
        adapter._executable = "/opt/llama-cli"
        with mock.patch("benchmarks.models.subprocess.run", return_value=mock.Mock(stdout="  \n")):
            with self.assertRaisesRegex(RuntimeError, "empty output"):
                adapter.generate("prompt")

    def test_llama_cli_permission_error_is_unavailable(self) -> None:
        adapter = LlamaCppAdapter({"path": "tiny.gguf"})
        adapter._loaded = True
        adapter._mode = "cli"
        adapter._executable = "/opt/llama-cli"
        with mock.patch("benchmarks.models.subprocess.run", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(BackendUnavailableError, "not executable"):
                adapter.generate("prompt")

    def test_explicit_non_executable_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "llama-cli"
            exe.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
            exe.chmod(0o644)
            with mock.patch("benchmarks.models.importlib.import_module", side_effect=ImportError):
                with self.assertRaisesRegex(BackendUnavailableError, "not executable"):
                    LlamaCppAdapter({"path": "tiny.gguf"}, executable=str(exe)).load()

    def test_generate_after_failed_load_is_unavailable(self) -> None:
        adapter = LlamaCppAdapter({"path": "tiny.gguf"})
        del adapter._mode
        with mock.patch.object(
            adapter, "load", side_effect=BackendUnavailableError("missing runtime")
        ):
            with self.assertRaises(BackendUnavailableError):
                adapter.generate("prompt")

    def test_generate_without_mode_is_unavailable(self) -> None:
        adapter = LlamaCppAdapter({"path": "tiny.gguf"})
        adapter._loaded = True
        adapter._mode = None
        with self.assertRaisesRegex(BackendUnavailableError, "did not finish loading"):
            adapter.generate("prompt")

    def test_does_not_treat_main_on_path_as_llama_cli(self) -> None:
        def fake_which(name: str) -> str | None:
            if name == "main":
                return "/usr/bin/main"
            return None

        with mock.patch("benchmarks.models.shutil.which", side_effect=fake_which):
            with mock.patch("benchmarks.models.importlib.import_module", side_effect=ImportError):
                with self.assertRaisesRegex(BackendUnavailableError, "llama-cli"):
                    LlamaCppAdapter({"path": "tiny.gguf"}).load()

    def test_vllm_backend_generates_text(self) -> None:
        class FakeLLM:
            def __init__(self, **kwargs):
                pass

            def generate(self, prompts, params, use_tqdm):
                return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="answer")])]

        module = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kwargs: kwargs)
        with mock.patch.dict(sys.modules, {"vllm": module}):
            adapter = VllmAdapter({"path": "tiny-model"})
            self.assertEqual(adapter.predict("question"), "answer")

    def test_missing_dependency_is_wrapped(self) -> None:
        with mock.patch("benchmarks.models.shutil.which", return_value=None):
            with mock.patch("benchmarks.models.importlib.import_module", side_effect=ImportError):
                with self.assertRaisesRegex(BackendUnavailableError, "llama-cpp-python"):
                    LlamaCppAdapter({"path": "tiny.gguf"}).load()


class PackageExportTests(unittest.TestCase):
    def test_load_manifest_is_exported(self) -> None:
        self.assertIs(exported_load_manifest, load_manifest)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text('{"runtime_format": "mock"}', encoding="utf-8")
            self.assertEqual(load_manifest(path)["runtime_format"], "mock")


class DetectBackendProducerShapesTest(unittest.TestCase):
    """`detect_backend` must accept what the Rust producers actually emit.

    These are literal producer-shaped dicts, not adapter-shaped fixtures. The
    previous implementation keyed off `runtime_format`, which nothing in this
    repository writes, so every one of these raised ValueError.
    """

    def test_run_manifest_checkpoint_format_gguf(self):
        # ExperimentManifest -> run_manifest.json
        manifest = {"checkpoint_format": "gguf", "checkpoint_path": "/models/m.gguf"}
        self.assertEqual(detect_backend(manifest), "llama.cpp")

    def test_run_manifest_checkpoint_format_safetensors(self):
        manifest = {"checkpoint_format": "safetensors", "checkpoint_path": "/models/dir"}
        self.assertEqual(detect_backend(manifest), "vllm")

    def test_run_matrix_source_format(self):
        self.assertEqual(
            detect_backend({"source_format": "gguf", "path": "/models/m.gguf"}),
            "llama.cpp",
        )

    def test_model_adapter_config_format_and_loader_hint(self):
        # ModelAdapterConfig carries `format` and `loader_hint`; every entry on
        # disk sets loader_hint to an artifact format, not a runtime name.
        entry = {
            "model_family": "olmoe",
            "model_id_or_local_path": "/models/olmoe",
            "format": "safetensors",
            "loader_hint": "safetensors",
        }
        self.assertEqual(detect_backend(entry), "vllm")

    def test_runtime_format_still_overrides_artifact_metadata(self):
        manifest = {
            "runtime_format": "vllm",
            "checkpoint_format": "gguf",
            "path": "/models/m.gguf",
        }
        self.assertEqual(detect_backend(manifest), "vllm")

    def test_custom_artifact_still_resolves_from_a_path_suffix(self):
        # Raising as soon as custom_artifact is seen would reject manifests
        # that previously resolved via the path.
        manifest = {"source_format": "custom_artifact", "path": "/models/m.gguf"}
        self.assertEqual(detect_backend(manifest), "llama.cpp")

    def test_sglang_is_named_as_unsupported(self):
        # Documented in the loader_hint domain of model_adapter_configs.toml
        # but has no adapter here.
        with self.assertRaises(ValueError) as ctx:
            detect_backend({"loader_hint": "sglang", "path": "/models/x"})
        self.assertIn("sglang", str(ctx.exception))
        self.assertIn("no adapter", str(ctx.exception))

    def test_unrecognised_values_are_echoed_in_the_error(self):
        with self.assertRaises(ValueError) as ctx:
            detect_backend({"source_format": "mystery", "path": "model.bin"})
        message = str(ctx.exception)
        self.assertIn("mystery", message)
        self.assertIn("source_format", message)

    def test_custom_artifact_gets_an_explanatory_error(self):
        with self.assertRaises(ValueError) as ctx:
            detect_backend({"source_format": "custom_artifact", "path": "/models/x"})
        message = str(ctx.exception)
        self.assertIn("custom_artifact", message)
        self.assertIn("runtime_format", message)
        # Actionable: names the values runtime_format accepts.
        for runtime in ("llama.cpp", "vllm", "huggingface"):
            self.assertIn(runtime, message)

    def test_custom_artifact_runtime_override_is_the_documented_fix(self):
        manifest = {"source_format": "custom_artifact", "path": "/models/x"}
        with self.assertRaises(ValueError):
            detect_backend(manifest)
        self.assertEqual(detect_backend({**manifest, "runtime_format": "vllm"}), "vllm")

    def test_unknown_metadata_names_the_fields_it_looked_at(self):
        with self.assertRaises(ValueError) as ctx:
            detect_backend({"path": "/models/mystery.bin.xz"})
        message = str(ctx.exception)
        self.assertIn("checkpoint_format", message)
        self.assertIn("runtime_format", message)


class HuggingFaceAdapterBehaviourTest(unittest.TestCase):
    """`HuggingFaceAdapter.load`/`generate` were only assertIsInstance-checked.

    Both `trust_remote_code` call sites and the prompt-stripping decode had no
    behavioural coverage, unlike the llama.cpp and vLLM adapters.
    """

    def test_trust_remote_code_reaches_both_from_pretrained_calls(self):
        calls = {}

        tokenizer = mock.MagicMock()
        encoded = mock.MagicMock()
        encoded.to.return_value = encoded
        encoded.__getitem__.return_value = [[1, 2, 3]]
        tokenizer.return_value = encoded
        tokenizer.decode.return_value = "the prompt and then the completion"
        tokenizer.eos_token_id = 0

        model = mock.MagicMock()
        model.generate.return_value = [[1, 2, 3, 4]]
        model.device = "cpu"

        def tok_from_pretrained(path, **kwargs):
            calls["tokenizer"] = kwargs
            return tokenizer

        def model_from_pretrained(path, **kwargs):
            calls["model"] = kwargs
            return model

        stub = types.ModuleType("transformers")
        stub.AutoTokenizer = mock.MagicMock()
        stub.AutoTokenizer.from_pretrained = tok_from_pretrained
        stub.AutoModelForCausalLM = mock.MagicMock()
        stub.AutoModelForCausalLM.from_pretrained = model_from_pretrained

        with mock.patch.dict(sys.modules, {"transformers": stub}):
            adapter = HuggingFaceAdapter(
                {"model_id_or_local_path": "/models/x", "format": "safetensors"},
                trust_remote_code=True,
            )
            adapter.load()

        self.assertTrue(calls["tokenizer"].get("trust_remote_code"))
        self.assertTrue(calls["model"].get("trust_remote_code"))

    def test_trust_remote_code_defaults_to_false_at_both_call_sites(self):
        calls = {}

        def tok_from_pretrained(path, **kwargs):
            calls["tokenizer"] = kwargs
            return _FakeHfTokenizer(prompt_ids=[])

        def model_from_pretrained(path, **kwargs):
            calls["model"] = kwargs
            return _FakeHfModel(output=[])

        stub = types.ModuleType("transformers")
        stub.AutoTokenizer = types.SimpleNamespace(from_pretrained=tok_from_pretrained)
        stub.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=model_from_pretrained)
        with mock.patch.dict(sys.modules, {"transformers": stub}):
            HuggingFaceAdapter({"model_id_or_local_path": "/models/x"}).load()
        self.assertIs(calls["tokenizer"]["trust_remote_code"], False)
        self.assertIs(calls["model"]["trust_remote_code"], False)

    def _loaded_adapter(self, model: _FakeHfModel, tokenizer: _FakeHfTokenizer, **options):
        stub = types.ModuleType("transformers")
        stub.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path, **kw: tokenizer)
        stub.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda path, **kw: model)
        with mock.patch.dict(sys.modules, {"transformers": stub}):
            adapter = HuggingFaceAdapter({"model_id_or_local_path": "/models/x"}, **options)
            adapter.load()
        return adapter

    def test_generate_moves_inputs_to_model_device_and_strips_the_prompt(self):
        tokenizer = _FakeHfTokenizer(prompt_ids=[11, 12, 13])
        model = _FakeHfModel(output=[11, 12, 13, 21, 22], device="cuda:0")
        adapter = self._loaded_adapter(model, tokenizer)

        text = adapter.generate("question", max_tokens=8)

        self.assertEqual(tokenizer.encoded.moved_to, "cuda:0")
        # Only the tokens after the prompt are decoded.
        self.assertEqual(tokenizer.decoded, [([21, 22], True)])
        self.assertEqual(text, "decoded:21,22")
        kwargs = model.generate_kwargs
        self.assertEqual(kwargs["max_new_tokens"], 8)
        self.assertEqual(kwargs["input_ids"].ids, [11, 12, 13])

    def test_generate_is_greedy_at_zero_temperature(self):
        tokenizer = _FakeHfTokenizer(prompt_ids=[1])
        model = _FakeHfModel(output=[1, 2])
        adapter = self._loaded_adapter(model, tokenizer)

        adapter.generate("q")

        kwargs = model.generate_kwargs
        self.assertFalse(kwargs["do_sample"])
        # transformers warns when temperature != 1.0 with do_sample=False.
        self.assertEqual(kwargs["temperature"], 1.0)

    def test_generate_samples_and_forwards_top_p_above_zero_temperature(self):
        tokenizer = _FakeHfTokenizer(prompt_ids=[1])
        model = _FakeHfModel(output=[1, 2])
        adapter = self._loaded_adapter(model, tokenizer)

        adapter.generate("q", temperature=0.7, top_p=0.9)

        kwargs = model.generate_kwargs
        self.assertTrue(kwargs["do_sample"])
        self.assertEqual(kwargs["temperature"], 0.7)
        self.assertEqual(kwargs["top_p"], 0.9)

    def test_generate_skips_device_move_for_a_parameterless_model(self):
        tokenizer = _FakeHfTokenizer(prompt_ids=[1])
        model = _FakeHfModel(output=[1, 2], device=None)
        adapter = self._loaded_adapter(model, tokenizer)

        adapter.generate("q")

        self.assertIsNone(tokenizer.encoded.moved_to)

    def test_unsupported_from_pretrained_options_are_dropped_with_a_warning(self):
        captured = {}
        stub = types.ModuleType("transformers")
        stub.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path, **kw: None)

        def model_from_pretrained(path, **kw):
            captured.update(kw)
            return _FakeHfModel([])

        stub.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=model_from_pretrained)
        with mock.patch.dict(sys.modules, {"transformers": stub}):
            adapter = HuggingFaceAdapter(
                {"model_id_or_local_path": "/models/x"},
                torch_dtype="bfloat16",
                not_a_real_option=1,
            )
            with self.assertWarnsRegex(UserWarning, "not_a_real_option"):
                adapter.load()
        self.assertEqual(captured, {"trust_remote_code": False, "torch_dtype": "bfloat16"})

    def test_missing_transformers_is_unavailable(self):
        with mock.patch("benchmarks.models.importlib.import_module", side_effect=ImportError):
            with self.assertRaisesRegex(BackendUnavailableError, "transformers"):
                HuggingFaceAdapter({"model_id_or_local_path": "/models/x"}).load()


class _FakeIds:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids
        self.shape = (1, len(ids))


class _FakeEncoding(dict):
    """Mimics transformers.BatchEncoding: a mapping with a ``.to(device)``."""

    def __init__(self, ids: list[int]) -> None:
        super().__init__(input_ids=_FakeIds(ids))
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class _FakeHfTokenizer:
    def __init__(self, prompt_ids: list[int]) -> None:
        self.prompt_ids = prompt_ids
        self.encoded: _FakeEncoding | None = None
        self.decoded: list[tuple[list[int], bool]] = []

    def __call__(self, prompt, return_tensors):
        self.encoded = _FakeEncoding(self.prompt_ids)
        return self.encoded

    def decode(self, ids, skip_special_tokens):
        self.decoded.append((list(ids), skip_special_tokens))
        return "decoded:" + ",".join(str(i) for i in ids)


class _FakeHfModel:
    def __init__(self, output: list[int], device: str | None = "cpu") -> None:
        self.output = output
        self.device = device
        self.generate_kwargs: dict | None = None

    def parameters(self):
        if self.device is None:
            return iter(())
        return iter([types.SimpleNamespace(device=self.device)])

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [self.output]


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER_CONFIGS = _REPO_ROOT / "configs" / "model_adapter_configs.toml"
_KNOWN_BACKENDS = {"llama.cpp", "vllm", "huggingface", "mock"}


class ModelAdapterConfigDetectionTest(unittest.TestCase):
    """Every `[[adapter]]` entry on disk must resolve or fail specifically.

    `configs/model_adapter_configs.toml` is the real producer of `format` and
    `loader_hint`. Before GH#182 all 20 entries raised the generic
    "cannot detect inference backend"; this drives the actual file, so a new
    entry with an unsupported value fails here with a message naming it.
    """

    @classmethod
    def setUpClass(cls) -> None:
        with _ADAPTER_CONFIGS.open("rb") as handle:
            cls.entries = tomllib.load(handle).get("adapter", [])

    def test_file_has_adapter_entries(self):
        self.assertGreater(len(self.entries), 0, f"no [[adapter]] entries in {_ADAPTER_CONFIGS}")

    def test_every_entry_resolves_or_names_its_unsupported_value(self):
        for entry in self.entries:
            family = entry.get("model_family", "<unnamed>")
            with self.subTest(model_family=family):
                try:
                    backend = detect_backend(entry)
                except ValueError as error:
                    message = str(error)
                    self.assertNotIn("cannot detect inference backend", message)
                    offending = {entry.get("format"), entry.get("loader_hint")} - {None}
                    self.assertTrue(
                        any(value in message for value in offending),
                        f"{family}: error does not name {offending}: {message}",
                    )
                    self.assertIn("runtime_format", message)
                else:
                    self.assertIn(backend, _KNOWN_BACKENDS)

    def test_entries_resolve_to_the_runtime_their_format_implies(self):
        expected = {"gguf": "llama.cpp", "safetensors": "vllm"}
        for entry in self.entries:
            if entry.get("format") not in expected or entry.get("loader_hint") != entry["format"]:
                continue
            with self.subTest(model_family=entry.get("model_family")):
                self.assertEqual(detect_backend(entry), expected[entry["format"]])

    def test_documented_value_domain_resolves_or_fails_specifically(self):
        # The header of model_adapter_configs.toml documents these domains:
        #   format       "gguf" | "safetensors" | "custom_artifact"
        #   loader_hint  "gguf" | "safetensors" | "vllm" | "sglang"
        resolves = {
            ("format", "gguf"): "llama.cpp",
            ("format", "safetensors"): "vllm",
            ("loader_hint", "gguf"): "llama.cpp",
            ("loader_hint", "safetensors"): "vllm",
            ("loader_hint", "vllm"): "vllm",
        }
        for (field, value), backend in resolves.items():
            with self.subTest(field=field, value=value):
                entry = {"model_id_or_local_path": "org/model", field: value}
                self.assertEqual(detect_backend(entry), backend)
        for field, value in (("format", "custom_artifact"), ("loader_hint", "sglang")):
            with self.subTest(field=field, value=value):
                entry = {"model_id_or_local_path": "org/model", field: value}
                with self.assertRaises(ValueError) as ctx:
                    detect_backend(entry)
                message = str(ctx.exception)
                self.assertIn(f"{field} ", message)
                self.assertIn(value, message)
                self.assertIn("runtime_format", message)


class LlamaCppLoadedStateChecksTest(unittest.TestCase):
    """The two former `assert`s in `LlamaCppAdapter.generate` are real checks.

    `python -O` strips asserts, which would have turned the documented
    BackendUnavailableError into an AttributeError/TypeError on the next line.
    """

    def test_python_mode_without_a_model_is_unavailable(self):
        adapter = LlamaCppAdapter({"path": "tiny.gguf"})
        adapter._loaded = True
        adapter._mode = "python"
        adapter._model = None
        with self.assertRaisesRegex(BackendUnavailableError, "holds no model"):
            adapter.generate("prompt")

    def test_cli_mode_without_an_executable_is_unavailable(self):
        adapter = LlamaCppAdapter({"path": "tiny.gguf"})
        adapter._loaded = True
        adapter._mode = "cli"
        adapter._executable = None
        with mock.patch("benchmarks.models.subprocess.run") as run:
            with self.assertRaisesRegex(BackendUnavailableError, "holds no executable"):
                adapter.generate("prompt")
        run.assert_not_called()

    def test_checks_survive_python_optimize(self):
        script = "\n".join(
            [
                "import sys",
                "from benchmarks.models import BackendUnavailableError, LlamaCppAdapter",
                "assert False  # stripped under -O; reaching the next line proves it",
                "failures = []",
                "for mode, attr in (('python', '_model'), ('cli', '_executable')):",
                "    adapter = LlamaCppAdapter({'path': 'tiny.gguf'})",
                "    adapter._loaded, adapter._mode = True, mode",
                "    setattr(adapter, attr, None)",
                "    try:",
                "        adapter.generate('prompt')",
                "    except BackendUnavailableError:",
                "        continue",
                "    except Exception as error:",
                "        failures.append(f'{mode}: {type(error).__name__}: {error}')",
                "    else:",
                "        failures.append(f'{mode}: no error')",
                "sys.exit('\\n'.join(failures) or 0)",
            ]
        )
        completed = subprocess.run(  # nosec B603
            [sys.executable, "-O", "-c", script],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
