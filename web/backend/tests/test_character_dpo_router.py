"""Character DPO router + service on a tmp dataset: list, import scan/apply
from a DPO concept built with ``export_single_pair``, finalize with the mask
step stubbed, queue build preview, and the image allow-list."""

import json
import os
import time

from modules.util import dpo_character_util as cu
from modules.util.dpo_curation_util import export_single_pair, load_manifest
from web.backend.routers.character_dpo import router
from web.backend.services.character_dpo_service import CharacterDPOService

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

NAME = "Rosie Butler"


def _png(path, color=(10, 20, 30), params: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (8, 8), color=color)
    if params is None:
        img.save(path)
    else:
        info = PngInfo()
        info.add_text("sui_image_params", json.dumps(params))
        img.save(path, pnginfo=info)
    return path


def _gen_params(prompt: str, loras=("RosieButler",)):
    return {"prompt": prompt, "model": "base", "loras": list(loras), "loraweights": [1.0] * len(loras)}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(CharacterDPOService, "_instance", None)
    # Never read the live queue.json or the active config from a test.
    monkeypatch.setattr(CharacterDPOService, "_queue_entries", staticmethod(list))
    monkeypatch.setattr(CharacterDPOService, "_global_concepts", staticmethod(list))
    monkeypatch.setattr(CharacterDPOService, "_global_workspace_dir", staticmethod(lambda: "ws/run"))
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "People"
    char = root / NAME
    for split, n in (("train", 3), ("val", 1)):
        for i in range(n):
            _png(str(char / split / f"{split}{i}.png"), (i, i, i))
            (char / split / f"{split}{i}.txt").write_text(f"{NAME} photo", encoding="utf-8")
            _png(str(char / split / f"{split}{i}-masklabel.png"), (255, 255, 255))
    _png(str(char / "dpo" / "generated" / "g0.png"), params=_gen_params("x"))
    (root / "Nobody").mkdir()  # no train/ → not a character
    models = tmp_path / "models"
    models.mkdir()
    (models / "RosieButler-patience-second.safetensors").write_bytes(b"second")
    return {"root": str(root), "models": str(models), "char": str(char)}


@pytest.fixture
def source_concept(tmp_path):
    """A shared DPO concept with 4 pairs: photo-chosen by name, AI-chosen by
    LoRA, one matching both, one unrelated."""
    concept_dir = str(tmp_path / "shared_dpo")
    src = tmp_path / "shared_src"
    manifest = load_manifest(concept_dir)
    # name match in prompt, real photo chosen → offpolicy
    export_single_pair(
        concept_dir,
        manifest,
        _png(str(src / "p0.png"), (1, 2, 3)),
        _png(str(src / "g0.png"), (3, 2, 1), params=_gen_params(f"{NAME} at the beach", loras=())),
        f"{NAME} at the beach",
        "1:1",
    )
    # lora match only, AI chosen → aivai (provenance from the file)
    export_single_pair(
        concept_dir,
        manifest,
        _png(str(src / "g1a.png"), (5, 5, 5), params=_gen_params("a woman in a park")),
        _png(str(src / "g1b.png"), (6, 6, 6), params=_gen_params("a woman in a park")),
        "a woman in a park",
        "1:1",
    )
    # both: name in caption + lora in stored rejected_provenance
    export_single_pair(
        concept_dir,
        manifest,
        _png(str(src / "p2.jpg"), (7, 7, 7)),
        _png(str(src / "g2.png"), (8, 8, 8), params=_gen_params("rosiebutler portrait")),
        "rosiebutler portrait",
        "1:1",
    )
    manifest["pairs"][-1]["rejected_provenance"] = {"model": "base", "loras": ["RosieButler"], "loraweights": [1.0]}
    # unrelated
    export_single_pair(
        concept_dir,
        manifest,
        _png(str(src / "p3.png"), (9, 9, 9)),
        _png(str(src / "g3.png"), (10, 10, 10), params=_gen_params("someone else", loras=("Other",))),
        "someone else",
        "1:1",
    )
    from modules.util.dpo_curation_util import save_manifest

    save_manifest(concept_dir, manifest)
    concept_file = tmp_path / "concepts.json"
    concept_file.write_text(
        json.dumps(
            [
                {
                    "name": "Shared DPO",
                    "path": concept_dir,
                    "enabled": True,
                    "include_subdirectories": True,
                    "dpo_chosen_pattern": "chosen/{}",
                    "dpo_rejected_pattern": "rejected/{}",
                },
                {"name": "Disabled", "path": concept_dir, "enabled": False, "dpo_chosen_pattern": "chosen/{}"},
            ]
        ),
        encoding="utf-8",
    )
    return {"dir": concept_dir, "file": str(concept_file)}


def test_list_characters(client, dataset):
    data = client.get(
        "/api/dpo/character/list", params={"root": dataset["root"], "models_dir": dataset["models"]}
    ).json()
    assert data["root"] == dataset["root"] and data["models_dir"] == dataset["models"]
    assert [c["name"] for c in data["characters"]] == [NAME]
    c = data["characters"][0]
    assert c["train_count"] == 3 and c["val_count"] == 1 and c["generated_count"] == 1
    assert c["offpolicy"] == {"train": 0, "val": 0, "unsplit": 0, "missing_masks": 0}
    assert c["aivai"] == {"train": 0, "val": 0, "unsplit": 0, "missing_masks": 0}
    assert c["lora"]["kind"] == "second" and c["lora"]["path"].endswith("RosieButler-patience-second.safetensors")
    assert c["lora"]["dir"] == dataset["models"]
    assert c["lora"]["swarm_name"] is None
    assert c["queue_entry"] is None and c["finalized"] is False and c["concepts_file"] is None


def test_models_dir_is_a_folder_list_searched_in_order(client, dataset, tmp_path, monkeypatch):
    """``models_dir`` on every route is ``;``-separated: a best in the second
    folder beats the second-best in the first, and the SwarmUI name keeps
    its folder prefix."""
    finished = tmp_path / "SoRealPeople!"
    finished.mkdir()
    (finished / "RosieButler.safetensors").write_bytes(b"finished")
    dirs = f"{dataset['models']};{finished}"
    import modules.util.swarm_client as swarm_client

    class Listing:
        def __init__(self, *a, **k):
            pass

        def list_models(self, subtype="Stable-Diffusion", depth=3):
            return ["SoRealPeople!/RosieButler", "Other/RosieButler-patience-second"]

    monkeypatch.setattr(swarm_client, "SwarmClient", Listing)
    data = client.get(
        "/api/dpo/character/list", params={"root": dataset["root"], "models_dir": dirs, "swarm_url": "http://s"}
    ).json()
    assert data["models_dir"] == dirs
    lora = data["characters"][0]["lora"]
    assert lora["kind"] == "best" and lora["dir"] == str(finished)
    assert lora["path"] == os.path.join(str(finished), "RosieButler.safetensors")
    assert lora["swarm_name"] == "SoRealPeople!/RosieButler"

    # Blank falls back to the default list (repo models first).
    from web.backend.services.character_dpo_service import DEFAULT_MODELS_DIR

    assert DEFAULT_MODELS_DIR.split(";")[1] == "E:/AI/Data/Models/Lora/SoRealPeople!"
    blank = client.get("/api/dpo/character/list", params={"root": dataset["root"], "models_dir": ""}).json()
    assert blank["models_dir"] == DEFAULT_MODELS_DIR

    # Publish: the LoRA already lives under the SwarmUI folder -> no copy.
    swarm_dir = tmp_path  # SoRealPeople! is a subfolder of it
    body = {"name": NAME, "root": dataset["root"], "models_dir": dirs, "swarm_lora_dir": str(swarm_dir)}
    published = client.post("/api/dpo/character/lora/publish", json=body).json()
    assert published == {"ok": True, "unchanged": True, "path": os.path.join(str(finished), "RosieButler.safetensors")}
    assert not os.path.exists(swarm_dir / "RosieButler.safetensors")

    # Queue build resolves through the same list and writes the DPO model beside it.
    sft = [
        {"name": f"Train - {NAME}", "path": dataset["char"] + "/train", "enabled": True, "type": "STANDARD"},
        {"name": f"Validation - {NAME}", "path": dataset["char"] + "/val", "enabled": True, "type": "VALIDATION"},
    ]
    monkeypatch.setattr(CharacterDPOService, "_global_concepts", staticmethod(lambda: sft))
    os.makedirs(cu.dpo_dir(dataset["root"], NAME), exist_ok=True)
    cu.write_character_concepts(
        dataset["root"], NAME, {"offpolicy": {"train": 1, "val": 0}, "aivai": {"train": 0, "val": 0}}
    )
    preview = client.post(
        "/api/dpo/character/queue/build", json={"names": [NAME], "root": dataset["root"], "models_dir": dirs}
    ).json()
    o = preview["entries"][0]["overrides"]
    assert o["lora_model_name"] == lora["path"]
    assert o["output_model_destination"] == os.path.join(str(finished), "RosieButler-dpo.safetensors").replace(
        "\\", "/"
    )


def test_list_reports_matching_queue_entry(client, dataset, monkeypatch):
    from modules.util.config.QueueConfig import QueueEntry

    entries = [QueueEntry(name="Other"), QueueEntry(name="rosie butler (copy)"), QueueEntry(name=NAME)]
    monkeypatch.setattr(CharacterDPOService, "_queue_entries", staticmethod(lambda: entries))
    data = client.get(
        "/api/dpo/character/list", params={"root": dataset["root"], "models_dir": dataset["models"]}
    ).json()
    assert data["characters"][0]["queue_entry"] == "rosie butler (copy)"


def test_list_missing_root_is_empty(client, tmp_path):
    data = client.get("/api/dpo/character/list", params={"root": str(tmp_path / "nope")}).json()
    assert data["characters"] == []


def test_import_scan_and_apply(client, dataset, source_concept):
    body = {
        "name": NAME,
        "root": dataset["root"],
        "concept_files": [source_concept["file"]],
        "models_dir": dataset["models"],
    }
    scan = client.post("/api/dpo/character/import/scan", json=body).json()
    by_id = {c["pair_id"]: c for c in scan["candidates"]}
    assert sorted(by_id) == [0, 1, 2]
    assert by_id[0]["match"] == "name" and by_id[0]["kind"] == "offpolicy"
    assert by_id[1]["match"] == "lora" and by_id[1]["kind"] == "aivai"
    assert by_id[2]["match"] == "both" and by_id[2]["kind"] == "offpolicy"
    assert by_id[0]["key"] == f"{source_concept['dir']}|0"
    assert by_id[0]["concept_name"] == "Shared DPO" and by_id[0]["caption"] == f"{NAME} at the beach"
    assert all(not c["already_imported"] for c in scan["candidates"])
    assert os.path.isfile(by_id[0]["chosen_path"]) and os.path.isfile(by_id[0]["rejected_path"])

    # Scanned concept dirs and the character root are served; anything else is 403.
    assert client.get("/api/dpo/character/image", params={"path": by_id[0]["chosen_path"]}).status_code == 200
    outside = _png(os.path.join(os.path.dirname(dataset["root"]), "outside.png"))
    assert client.get("/api/dpo/character/image", params={"path": outside}).status_code == 403

    picks = [{"concept_dir": source_concept["dir"], "pair_id": pid} for pid in (0, 1, 2)]
    applied = client.post(
        "/api/dpo/character/import/apply", json={"name": NAME, "root": dataset["root"], "picks": picks}
    ).json()
    assert applied == {"ok": True, "imported": 3, "skipped": 0, "errors": []}

    off = load_manifest(cu.pair_dir(dataset["root"], NAME, "offpolicy"))
    ai = load_manifest(cu.pair_dir(dataset["root"], NAME, "aivai"))
    assert [e["source_pair_id"] for e in off["pairs"]] == [0, 2]
    assert [e["source_pair_id"] for e in ai["pairs"]] == [1]
    assert off["pairs"][1]["rejected_provenance"]["loras"] == ["RosieButler"]
    assert off["pairs"][0]["source_concept"] == source_concept["dir"]
    off_dir = cu.pair_dir(dataset["root"], NAME, "offpolicy")
    assert os.path.isfile(os.path.join(off_dir, "chosen", "pair_0001.jpg"))
    with open(os.path.join(off_dir, "chosen", "pair_0000.txt"), encoding="utf-8") as f:
        assert f.read() == f"{NAME} at the beach"
    state = cu.load_character_state(dataset["root"], NAME)
    assert len(state["imported"]) == 3

    # Second apply of the same picks is skipped; the scan now reports them imported.
    again = client.post(
        "/api/dpo/character/import/apply", json={"name": NAME, "root": dataset["root"], "picks": picks}
    ).json()
    assert again["imported"] == 0 and again["skipped"] == 3
    scan = client.post("/api/dpo/character/import/scan", json=body).json()
    assert all(c["already_imported"] for c in scan["candidates"])

    listed = client.get(
        "/api/dpo/character/list", params={"root": dataset["root"], "models_dir": dataset["models"]}
    ).json()
    assert listed["characters"][0]["offpolicy"] == {"train": 0, "val": 0, "unsplit": 2, "missing_masks": 4}
    assert listed["characters"][0]["aivai"]["unsplit"] == 1


def _wait_finalize(client, root, name, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get("/api/dpo/character/finalize/status", params={"name": name, "root": root}).json()
        if not status["running"]:
            return status
        time.sleep(0.05)
    raise AssertionError("finalize did not finish")


def test_finalize_and_queue_preview(client, dataset, source_concept, monkeypatch):
    root = dataset["root"]
    body = {"name": NAME, "root": root, "concept_files": [source_concept["file"]], "models_dir": dataset["models"]}
    client.post("/api/dpo/character/import/scan", json=body)
    picks = [{"concept_dir": source_concept["dir"], "pair_id": pid} for pid in (0, 1, 2)]
    client.post("/api/dpo/character/import/apply", json={"name": NAME, "root": root, "picks": picks}).json()
    # The real photo behind pair 0 has a mask next to its source: it is copied, not generated.
    _png(cu.mask_path_for(os.path.join(os.path.dirname(source_concept["dir"]), "shared_src", "p0.png")), (255, 0, 0))

    generated: list[str] = []

    def fake_masks(self, paths, mask_model, mask_options, job):
        assert mask_model == "Rembg-Human"
        for path in paths:
            _png(cu.mask_path_for(path), (255, 255, 255))
            generated.append(path)
        return len(paths)

    monkeypatch.setattr(CharacterDPOService, "_generate_masks", fake_masks)

    started = client.post(
        "/api/dpo/character/finalize", json={"name": NAME, "root": root, "val_pct": 50, "dpo_resolution": "896"}
    ).json()
    assert started["ok"] and started["job_id"]
    status = _wait_finalize(client, root, NAME)
    assert status["error"] is None, status
    result = status["result"]
    assert result["offpolicy"] == {"train": 1, "val": 1}
    assert result["aivai"] == {"train": 1, "val": 0}
    with open(result["concepts_file"], encoding="utf-8") as f:
        written = json.load(f)
    assert {c["image"]["resolution_override"] for c in written} == {"896"}
    assert all(c["image"]["enable_resolution_override"] for c in written)
    assert cu.load_character_state(root, NAME)["last_finalize"]["dpo_resolution"] == "896"
    assert result["masks_copied"] == 1 and result["masks_generated"] == 5
    assert len(generated) == 5
    assert os.path.isfile(result["concepts_file"])
    with open(result["concepts_file"], encoding="utf-8") as f:
        concepts = json.load(f)
    assert [c["enabled"] for c in concepts] == [True, True, True, False]

    listed = client.get("/api/dpo/character/list", params={"root": root, "models_dir": dataset["models"]}).json()
    c = listed["characters"][0]
    assert c["finalized"] and c["concepts_file"] == result["concepts_file"]
    assert c["offpolicy"] == {"train": 1, "val": 1, "unsplit": 0, "missing_masks": 0}
    assert c["aivai"] == {"train": 1, "val": 0, "unsplit": 0, "missing_masks": 0}
    state = cu.load_character_state(root, NAME)
    assert state["last_finalize"]["masks_copied"] == 1

    # Queue preview: no queue entry exists, so SFT concepts come from the global config.
    from web.backend.services import character_dpo_service as svc

    sft = [
        {"name": f"Train - {NAME}", "path": dataset["char"] + "/train", "enabled": True, "type": "STANDARD"},
        {"name": f"Validation - {NAME}", "path": dataset["char"] + "/val", "enabled": True, "type": "VALIDATION"},
    ]
    monkeypatch.setattr(svc.CharacterDPOService, "_global_concepts", staticmethod(lambda: sft))
    preview = client.post(
        "/api/dpo/character/queue/build",
        json={"names": [NAME, "Nobody"], "root": root, "models_dir": dataset["models"], "add_on_weight": 0.6},
    ).json()
    assert preview["added"] == 0
    assert [e["name"] for e in preview["errors"]] == ["Nobody"]
    assert len(preview["entries"]) == 1
    entry = preview["entries"][0]
    assert entry["name"] == f"{NAME} DPO"
    o = entry["overrides"]
    assert o["rlhf_mode"] == "SFT_DPO" and o["rlhf_dpo_addon_weight"] == 0.6
    assert o["rlhf_dpo_validation"] is True
    assert o["workspace_dir"] == "ws/run/dpo/RosieButler"
    assert len(o["concepts"]) == 6
    assert any("No SFT queue entry" in w for w in entry["warnings"])

    # mode "add" goes through the queue service.
    added_entries = []

    class FakeQueue:
        def add_entry(self, name, overrides):
            added_entries.append((name, overrides))
            return {"id": "x"}

    import web.backend.services.queue_service as qs

    monkeypatch.setattr(qs.QueueService, "get_instance", classmethod(lambda cls: FakeQueue()))
    added = client.post(
        "/api/dpo/character/queue/build",
        json={"names": [NAME], "root": root, "models_dir": dataset["models"], "mode": "add"},
    ).json()
    assert added["added"] == 1 and added_entries[0][0] == f"{NAME} DPO"


def test_finalize_conflict_and_missing_character(client, dataset, monkeypatch):
    import threading

    release = threading.Event()

    def slow_masks(self, paths, mask_model, mask_options, job):
        release.wait(5)
        return 0

    monkeypatch.setattr(CharacterDPOService, "_generate_masks", slow_masks)
    root = dataset["root"]
    folder = cu.pair_dir(root, NAME, "aivai")
    manifest = load_manifest(folder)
    src = os.path.join(root, ".src")
    export_single_pair(folder, manifest, _png(os.path.join(src, "a.png")), _png(os.path.join(src, "b.png")), "p", "1:1")
    first = client.post("/api/dpo/character/finalize", json={"name": NAME, "root": root})
    assert first.status_code == 200 and first.json()["ok"]
    second = client.post("/api/dpo/character/finalize", json={"name": NAME, "root": root})
    assert second.status_code == 409
    release.set()
    status = _wait_finalize(client, root, NAME)
    assert status["error"] is None
    missing = client.post("/api/dpo/character/finalize", json={"name": "Ghost", "root": root}).json()
    assert missing["ok"] is False


def test_train_captions(client, dataset):
    data = client.get("/api/dpo/character/train-captions", params={"name": NAME, "root": dataset["root"]}).json()
    assert data == {"count": 3}
    assert client.get(
        "/api/dpo/character/train-captions", params={"name": "Ghost", "root": dataset["root"]}
    ).json() == {"count": 0}


class FakeGenerate:
    """Stands in for SwarmGenerateService: records every ``start`` call."""

    calls: list[dict] = []
    fail_for: str | None = None

    def start(self, **kwargs):
        FakeGenerate.calls.append(kwargs)
        if FakeGenerate.fail_for and kwargs.get("label") == FakeGenerate.fail_for:
            return {"ok": False, "error": "Real photo folder not found"}
        return {"ok": True, "queued": len(FakeGenerate.calls) > 1, "position": len(FakeGenerate.calls)}


@pytest.fixture
def swarm(monkeypatch, tmp_path):
    """A fake SwarmUI: LoRA listing (scripted per call) and the generate service."""
    import modules.util.swarm_client as swarm_client
    import web.backend.services.character_dpo_service as svc
    import web.backend.services.swarm_generate_service as gen_module

    listings: list[list[str]] = []
    list_calls: list[tuple[str, str | None]] = []

    class FakeSwarmClient:
        def __init__(self, base_url, token=None, timeout=None):
            list_calls.append((base_url, token))

        def list_models(self, subtype="Stable-Diffusion", depth=3):
            assert subtype == "LoRA"
            return listings.pop(0) if listings else []

    monkeypatch.setattr(swarm_client, "SwarmClient", FakeSwarmClient)
    monkeypatch.setattr(svc, "SWARM_LIST_RETRY_DELAY", 0.0)
    FakeGenerate.calls = []
    FakeGenerate.fail_for = None
    monkeypatch.setattr(gen_module.SwarmGenerateService, "get_instance", classmethod(lambda cls: FakeGenerate()))
    lora_dir = tmp_path / "swarm_loras"
    lora_dir.mkdir()
    reference = tmp_path / "Reference"
    _png(str(reference / "ref.png"), params={"model": "base", "steps": 8, "loras": ["Other"], "loraweights": ["1"]})
    return {"listings": listings, "list_calls": list_calls, "lora_dir": str(lora_dir), "reference": str(reference)}


def _generate_body(dataset, swarm, names, **extra):
    return {
        "names": names,
        "root": dataset["root"],
        "models_dir": dataset["models"],
        "swarm_url": "http://swarm:7801",
        "swarm_token": "tok",
        "swarm_lora_dir": swarm["lora_dir"],
        "param_folder": swarm["reference"],
        **extra,
    }


def test_generate_queues_one_run_per_character_with_its_own_lora(client, dataset, swarm):
    root = dataset["root"]
    # A second character with captions but no LoRA: reported, not fatal.
    _png(os.path.join(root, "No Lora", "train", "t0.png"))
    with open(os.path.join(root, "No Lora", "train", "t0.txt"), "w", encoding="utf-8") as f:
        f.write("no lora photo")
    # SwarmUI lists the LoRA only on the second ask (its folder scan lags the copy).
    swarm["listings"].extend([[], ["Chars/RosieButler-patience-second.safetensors"]])

    data = client.post("/api/dpo/character/generate", json=_generate_body(dataset, swarm, ["No Lora", NAME])).json()
    assert data["ok"] is False
    by_name = {job["name"]: job for job in data["jobs"]}
    assert by_name["No Lora"]["ok"] is False and "No LoRA" in by_name["No Lora"]["error"]
    rosie = by_name[NAME]
    assert rosie["ok"] is True, rosie
    assert rosie["position"] == 1 and rosie["swarm_lora"] == "Chars/RosieButler-patience-second.safetensors"
    assert os.path.isfile(os.path.join(swarm["lora_dir"], "RosieButler-patience-second.safetensors"))
    assert swarm["list_calls"] == [("http://swarm:7801", "tok")] * 2

    (call,) = FakeGenerate.calls
    assert call["real_folder"] == os.path.join(dataset["char"], "train")
    assert call["generated_folder"] == os.path.join(dataset["char"], "dpo", "generated")
    assert call["output_dir"] == os.path.join(dataset["char"], "dpo", "offpolicy")
    assert os.path.isdir(call["generated_folder"])
    assert call["model"] == "" and call["base_url"] == "http://swarm:7801" and call["token"] == "tok"
    assert call["captions"] == 3 and call["images_per_caption"] == 2
    assert call["param_source"] == "folder" and call["param_folder"] == swarm["reference"]
    assert call["drop_source_loras"] is True
    assert call["loras"] == [{"name": "Chars/RosieButler-patience-second.safetensors", "weight": 1.0}]
    assert call["label"] == NAME
    assert call["save_to"] == call["generated_folder"]
    assert call["concurrency"] == 3


def test_generate_reports_swarm_and_queue_failures_per_character(client, dataset, swarm):
    root = dataset["root"]
    models = dataset["models"]
    # Two characters with LoRAs: the first never shows up in SwarmUI, the
    # second is refused by the generate service; neither stops the other.
    for name, stem in (("Ann Other", "AnnOther"), ("Bea Third", "BeaThird")):
        _png(os.path.join(root, name, "train", "t0.png"))
        with open(os.path.join(root, name, "train", "t0.txt"), "w", encoding="utf-8") as f:
            f.write(f"{name} photo")
        with open(os.path.join(models, f"{stem}.safetensors"), "wb") as f:
            f.write(stem.encode())
    swarm["listings"].extend(
        [[], [], ["BeaThird.safetensors"], ["BeaThird.safetensors", "RosieButler-patience-second"]]
    )
    FakeGenerate.fail_for = "Bea Third"

    body = _generate_body(dataset, swarm, ["Ann Other", "Bea Third", NAME], captions=2, images_per_caption=1)
    body["concurrency"] = 5
    data = client.post("/api/dpo/character/generate", json=body).json()
    by_name = {job["name"]: job for job in data["jobs"]}
    assert data["ok"] is False
    assert "not in SwarmUI's LoRA list" in by_name["Ann Other"]["error"]
    assert by_name["Bea Third"]["ok"] is False and by_name["Bea Third"]["error"] == "Real photo folder not found"
    assert by_name["Bea Third"]["swarm_lora"] == "BeaThird.safetensors"
    assert by_name[NAME]["ok"] is True and by_name[NAME]["position"] == 2
    assert [c["label"] for c in FakeGenerate.calls] == ["Bea Third", NAME]
    assert all(
        c["captions"] == 2 and c["images_per_caption"] == 1 and c["concurrency"] == 5 for c in FakeGenerate.calls
    )


def test_generate_refuses_a_reference_folder_without_a_generation(client, dataset, swarm, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    data = client.post(
        "/api/dpo/character/generate", json=_generate_body(dataset, swarm, [NAME], param_folder=str(empty))
    ).json()
    assert data["ok"] is False and data["jobs"] == [] and "parameter block" in data["error"]
    missing = client.post(
        "/api/dpo/character/generate", json=_generate_body(dataset, swarm, [NAME], param_folder=str(tmp_path / "x"))
    ).json()
    assert missing["ok"] is False and "not found" in missing["error"]
    assert FakeGenerate.calls == []


def _wait_prepare(client, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get("/api/dpo/character/prepare/status").json()
        if not status["running"]:
            return status
        time.sleep(0.05)
    raise AssertionError("prepare did not finish")


def test_prepare_finalizes_each_character_and_skips_finished_ones(client, dataset, monkeypatch):
    root = dataset["root"]
    idle = client.get("/api/dpo/character/prepare/status").json()
    assert idle == {
        "running": False,
        "current": None,
        "done": 0,
        "total": 0,
        "stage": "idle",
        "message": "",
        "error": None,
        "results": {},
    }

    # Rosie: one unsplit, unmasked AI-vs-AI pair -> needs the full finalize.
    folder = cu.pair_dir(root, NAME, "aivai")
    manifest = load_manifest(folder)
    src = os.path.join(root, ".src")
    export_single_pair(folder, manifest, _png(os.path.join(src, "a.png")), _png(os.path.join(src, "b.png")), "p", "1:1")
    # Done: concepts written, nothing unsplit or unmasked -> skipped.
    done = "Done Person"
    _png(os.path.join(root, done, "train", "t0.png"))
    os.makedirs(cu.dpo_dir(root, done), exist_ok=True)
    with open(os.path.join(cu.dpo_dir(root, done), cu.CONCEPTS_FILENAME), "w", encoding="utf-8") as f:
        f.write("[]")
    assert CharacterDPOService.needs_finalize(root, NAME) and not CharacterDPOService.needs_finalize(root, done)

    masked: list[str] = []

    def fake_masks(self, paths, mask_model, mask_options, job):
        assert mask_model == "Rembg-Human" and mask_options == {"expand": 4}
        for path in paths:
            _png(cu.mask_path_for(path), (255, 255, 255))
            masked.append(path)
        return len(paths)

    monkeypatch.setattr(CharacterDPOService, "_generate_masks", fake_masks)
    started = client.post(
        "/api/dpo/character/prepare",
        json={
            "names": [done, NAME, "Ghost"],
            "root": root,
            "val_pct": 0,
            "mask_options": {"expand": 4},
            "dpo_resolution": "1280",
        },
    ).json()
    assert started == {"ok": True, "total": 3}
    status = _wait_prepare(client)
    assert status["error"] is None and status["stage"] == "done" and status["current"] is None
    assert status["done"] == 3 and status["total"] == 3
    assert status["results"][done] == {"skipped": True}
    assert "not found" in status["results"]["Ghost"]["error"]
    result = status["results"][NAME]
    assert result["aivai"] == {"train": 1, "val": 0} and result["offpolicy"] == {"train": 0, "val": 0}
    assert result["masks_generated"] == 2 and len(masked) == 2
    assert os.path.isfile(result["concepts_file"])
    with open(result["concepts_file"], encoding="utf-8") as f:
        assert {c["image"]["resolution_override"] for c in json.load(f)} == {"1280"}
    # The per-character finalize status saw the same job.
    single = client.get("/api/dpo/character/finalize/status", params={"name": NAME, "root": root}).json()
    assert single["result"] == result and single["running"] is False
    # Now Rosie is finished too: a second batch skips both.
    assert not CharacterDPOService.needs_finalize(root, NAME)
    client.post("/api/dpo/character/prepare", json={"names": [NAME, done], "root": root})
    again = _wait_prepare(client)
    assert again["results"] == {NAME: {"skipped": True}, done: {"skipped": True}}
    assert client.post("/api/dpo/character/prepare", json={"names": [], "root": root}).json()["ok"] is False


def test_prepare_conflicts_with_itself_and_with_a_single_finalize(client, dataset, monkeypatch):
    import threading

    release = threading.Event()

    def slow_masks(self, paths, mask_model, mask_options, job):
        job["stage"] = "generating masks"
        release.wait(5)
        return 0

    monkeypatch.setattr(CharacterDPOService, "_generate_masks", slow_masks)
    root = dataset["root"]
    folder = cu.pair_dir(root, NAME, "aivai")
    manifest = load_manifest(folder)
    src = os.path.join(root, ".src")
    export_single_pair(folder, manifest, _png(os.path.join(src, "a.png")), _png(os.path.join(src, "b.png")), "p", "1:1")

    first = client.post("/api/dpo/character/prepare", json={"names": [NAME], "root": root})
    assert first.status_code == 200 and first.json()["ok"]
    deadline = time.time() + 5
    while time.time() < deadline:
        status = client.get("/api/dpo/character/prepare/status").json()
        if status["stage"] == "generating masks":
            break
        time.sleep(0.02)
    assert status["running"] and status["current"] == NAME and status["stage"] == "generating masks"
    assert client.post("/api/dpo/character/prepare", json={"names": [NAME], "root": root}).status_code == 409
    assert client.post("/api/dpo/character/finalize", json={"name": NAME, "root": root}).status_code == 409
    release.set()
    _wait_prepare(client)

    # And the other way round: a single finalize in flight blocks a batch.
    release.clear()
    assert client.post("/api/dpo/character/finalize", json={"name": NAME, "root": root}).status_code == 200
    assert client.post("/api/dpo/character/prepare", json={"names": [NAME], "root": root}).status_code == 409
    release.set()
    _wait_finalize(client, root, NAME)


def test_publish_lora(client, dataset, tmp_path):
    swarm = tmp_path / "swarm_loras"
    swarm.mkdir()
    body = {"name": NAME, "root": dataset["root"], "models_dir": dataset["models"], "swarm_lora_dir": str(swarm)}
    first = client.post("/api/dpo/character/lora/publish", json=body).json()
    assert first["ok"] and os.path.isfile(first["path"])
    assert os.path.basename(first["path"]) == "RosieButler-patience-second.safetensors"
    (swarm / "RosieButler-patience-second.safetensors").write_bytes(b"different")
    clash = client.post("/api/dpo/character/lora/publish", json=body).json()
    assert clash["ok"] is False and "different" in clash["error"]
    forced = client.post("/api/dpo/character/lora/publish", json=body | {"force": True}).json()
    assert forced["ok"]
    with open(forced["path"], "rb") as f:
        assert f.read() == b"second"
