from __future__ import annotations

import math
import sys
import tempfile
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


if __name__ == "__main__":
    unittest.main()


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

    def test_custom_artifact_gets_an_explanatory_error(self):
        with self.assertRaises(ValueError) as ctx:
            detect_backend({"source_format": "custom_artifact", "path": "/models/x"})
        self.assertIn("custom_artifact", str(ctx.exception))
        self.assertIn("runtime_format", str(ctx.exception))

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


