import importlib.util
import json
from pathlib import Path


def test_publication_excludes_optimizer_and_private_files(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/vln/upload_campaign.py"
    assert script.exists(), "Publication allowlist is missing"
    spec = importlib.util.spec_from_file_location("publication", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for epoch, step in enumerate((1353, 2706, 4059), 1):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        for name in (
            "pytorch_model.bin",
            "config.json",
            "navigation.json",
            "tokenizer.json",
        ):
            (checkpoint / name).write_text("test")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"epoch": epoch, "global_step": step})
        )
        (checkpoint / "optimizer.bin").write_text("private")
        (checkpoint / "token").write_text("private")
    files = module.publication_files(tmp_path)
    assert "epoch-1/pytorch_model.bin" in files
    assert "epoch-2/navigation.json" in files
    assert "epoch-3/tokenizer.json" in files
    assert not any("optimizer" in name or name.endswith("/token") for name in files)
