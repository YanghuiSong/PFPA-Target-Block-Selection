#!/usr/bin/env python3
"""Checkpoint-producing PFPA run using the frozen historical update helpers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(os.environ.get("PFPA_DATA_ROOT", Path.cwd())).expanduser().resolve()
RELEASE_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(RELEASE_SCRIPTS))
import probe_function_preserving_update as pf  # noqa: E402
import probe_transformer_block_plasticity as core  # noqa: E402

BLOCK_NAME = "stage1_block1"
EPSILON = 0.001
IMAGE_CAP = 128
IMAGE_SIZE = 512
BATCH_SIZE = 4
WORKERS = 4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("".join(x + "\n" for x in ids).encode("utf-8")).hexdigest()


def save_sample_ids(path: Path, frame: pd.DataFrame) -> str:
    ids = frame["sample_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate sample IDs in {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["manifest_row_after_seeded_selection", "sample_id"])
        for i, sid in enumerate(ids):
            w.writerow([i, sid])
    return file_ids_sha256(ids)


def validate_source_checkpoint_provenance(
    path: Path, source_domain_tag: str, source_manifest_sha256: str,
    checkpoint_sha256: str, split_seed: int,
) -> dict:
    provenance = json.loads(path.read_text(encoding="utf-8"))
    required = {"method", "source_domain", "source_train_manifest_sha256",
                "source_checkpoint_sha256", "split_seed"}
    missing = required - set(provenance)
    if missing:
        raise SystemExit(f"ABORT: source checkpoint provenance missing {sorted(missing)}")
    method = "".join(c for c in str(provenance["method"]).lower() if c.isalnum())
    if method != "sourceonly":
        raise SystemExit("ABORT: source checkpoint provenance method is not SourceOnly")
    normalize = lambda x: "".join(c for c in str(x).lower() if c.isalnum())
    tagged, recorded = normalize(source_domain_tag), normalize(provenance["source_domain"])
    if not tagged or not recorded or not (tagged in recorded or recorded in tagged):
        raise SystemExit("ABORT: source checkpoint provenance domain does not match source tag")
    if provenance["source_train_manifest_sha256"] != source_manifest_sha256:
        raise SystemExit("ABORT: checkpoint was not trained from this source_train manifest")
    if provenance["source_checkpoint_sha256"] != checkpoint_sha256:
        raise SystemExit("ABORT: checkpoint SHA256 does not match provenance")
    if int(provenance["split_seed"]) != int(split_seed):
        raise SystemExit("ABORT: source checkpoint provenance split seed mismatch")
    return provenance


def validate_manifest(path: Path, *, require_mask: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sample_id", "image"}
    if require_mask:
        required.add("mask")
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"ABORT: {path} missing required columns {sorted(missing)}")
    if frame.empty or frame["sample_id"].isna().any() or frame["image"].isna().any():
        raise SystemExit(f"ABORT: {path} has empty sample/image entries")
    ids = frame["sample_id"].astype(str)
    if ids.duplicated().any():
        raise SystemExit(f"ABORT: {path} contains duplicate sample_id values")
    if require_mask and (frame["mask"].isna().any() or (frame["mask"].astype(str).str.strip() == "").any()):
        raise SystemExit(f"ABORT: labeled source manifest {path} has missing masks")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="target/task label recorded in artifacts")
    ap.add_argument("--seed", required=True, type=int, choices=[2027, 3407, 4096])
    ap.add_argument("--source-domain-tag", required=True)
    ap.add_argument("--source-train-manifest", required=True)
    ap.add_argument("--source-val-manifest", required=True)
    ap.add_argument("--source-checkpoint", required=True)
    ap.add_argument("--source-checkpoint-provenance", required=True)
    ap.add_argument("--pretrained-dir", required=True,
                    help="Local Hugging Face MiT-B0 directory; downloads are disabled")
    ap.add_argument("--image-cap", type=int, default=IMAGE_CAP)
    ap.add_argument("--target-train", required=True)
    ap.add_argument("--target-val", required=True)
    ap.add_argument("--target-test", required=True)
    ap.add_argument("--proxy-matrix", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--protocol-sha256", required=True)
    ap.add_argument("--runner-sha256", default=None,
                    help="Optional expected SHA256; defaults to this script's current SHA256")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reserved = [output / x for x in ["adapted_checkpoint.pt", "config.yaml", "gradient_diagnostic.json"]]
    if any(p.exists() for p in reserved):
        raise SystemExit("ABORT: refusing to overwrite a confirmatory artifact")

    protocol_path = Path(args.protocol).resolve()
    actual_protocol_sha = sha256(protocol_path)
    if actual_protocol_sha != args.protocol_sha256:
        raise SystemExit("ABORT: PFPA protocol hash mismatch")

    source_checkpoint = Path(args.source_checkpoint).resolve()
    source_checkpoint_provenance = Path(args.source_checkpoint_provenance).resolve()
    pretrained = Path(args.pretrained_dir).expanduser().resolve()
    runner_sha256 = args.runner_sha256 or sha256(Path(__file__).resolve())
    source_train = Path(args.source_train_manifest).resolve()
    source_val = Path(args.source_val_manifest).resolve()
    target_train = Path(args.target_train).resolve()
    target_val = Path(args.target_val).resolve()
    target_test = Path(args.target_test).resolve()
    proxy_path = Path(args.proxy_matrix).resolve()
    selector_path = proxy_path.parent / "selection.json"
    if args.image_cap <= 0 or args.image_cap > IMAGE_CAP:
        raise SystemExit(f"ABORT: --image-cap must be within 1..{IMAGE_CAP}")
    if not all(p.exists() for p in [source_checkpoint, source_checkpoint_provenance, pretrained, source_train, source_val, target_train, target_val, target_test, proxy_path, selector_path]):
        raise SystemExit("ABORT: one or more frozen protocol inputs are missing")
    source_train_manifest = validate_manifest(source_train, require_mask=True)
    validate_manifest(source_val, require_mask=True)
    target_train_manifest = validate_manifest(target_train, require_mask=False)
    validate_manifest(target_val, require_mask=False)
    validate_manifest(target_test, require_mask=False)
    source_checkpoint_sha = sha256(source_checkpoint)
    source_manifest_sha = sha256(source_train)
    validated_source_provenance = validate_source_checkpoint_provenance(
        source_checkpoint_provenance, args.source_domain_tag,
        source_manifest_sha, source_checkpoint_sha, args.seed,
    )
    source_provenance_sha = sha256(source_checkpoint_provenance)

    selection = json.loads(selector_path.read_text(encoding="utf-8"))
    proxy = pd.read_csv(proxy_path)
    top = proxy.loc[proxy["target_potential"].idxmax()]
    selected_block = str(top["block"])
    if selected_block != BLOCK_NAME or selection.get("selected_block") != BLOCK_NAME:
        raise SystemExit(f"ABORT: selector differs from frozen historical block: {selected_block}")
    if bool(selection.get("uses_target_labels", True)):
        raise SystemExit("ABORT: selector evidence indicates target labels were used")

    target_ds = pf.TrainImages(target_train, IMAGE_SIZE, args.image_cap, args.seed, masks=False)
    source_ds = pf.TrainImages(source_train, IMAGE_SIZE, args.image_cap, args.seed, masks=True)
    target_ids = target_ds.df["sample_id"].astype(str).tolist()
    source_ids = source_ds.df["sample_id"].astype(str).tolist()
    if len(target_ids) != int(selection["calibration_images"]):
        raise SystemExit("ABORT: selector/adaptation target sample count mismatch")
    if len(target_ids) != min(args.image_cap, len(target_train_manifest)):
        raise SystemExit("ABORT: effective target sample budget mismatch")
    if len(source_ids) != min(args.image_cap, len(source_train_manifest)):
        raise SystemExit("ABORT: source-anchor sample budget mismatch")

    target_ids_sha = save_sample_ids(output / "target_calibration_sample_ids.csv", target_ds.df)
    source_ids_sha = save_sample_ids(output / "source_anchor_sample_ids.csv", source_ds.df)
    target_loader = DataLoader(target_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)
    source_loader = DataLoader(source_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("ABORT: confirmatory PFPA requires the allocated CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)
    gpu_start.record()

    stage, block_index = pf.parse_block(selected_block)
    target_model = pf.build_model(pretrained, source_checkpoint, device)
    target_block = pf.selected_block(target_model, stage, block_index)
    target_loss = pf.target_gradient(
        target_model, target_block, target_loader, device,
        entropy_weight=1.0, consistency_weight=0.5,
    )
    target_gradient = pf.flatten_grad(target_block).detach().cpu()
    del target_model
    torch.cuda.empty_cache()

    source_model = pf.build_model(pretrained, source_checkpoint, device)
    source_block = pf.selected_block(source_model, stage, block_index)
    source_loss = pf.source_gradient(source_model, source_block, source_loader, device)
    source_gradient = pf.flatten_grad(source_block).detach().cpu()
    del source_model
    torch.cuda.empty_cache()

    projected, projection = pf.project_conflict(target_gradient, source_gradient)
    model = pf.build_model(pretrained, source_checkpoint, device)
    block = pf.selected_block(model, stage, block_index)
    core.freeze_strict(model, block)
    module_name = next(name for name, module in model.named_modules() if module is block)
    selected_prefix = module_name + "."
    before_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    before_vector = core.parameter_vector(block).detach().double()
    trainable_params = sum(p.numel() for p in block.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    if trainable_params != 79616 or total_params != 3714915:
        raise SystemExit(f"ABORT: architecture parameter mismatch: {trainable_params}/{total_params}")

    update = pf.apply_normalized_update(block, projected.to(device), EPSILON)
    after_vector = core.parameter_vector(block).detach().double()
    measured_displacement = float((after_vector - before_vector).norm().item() / max(before_vector.norm().item(), 1e-12))
    state = model.state_dict()
    changed_keys = [k for k, v in state.items() if not torch.equal(before_state[k], v.detach().cpu())]
    if not changed_keys or any(not k.startswith(selected_prefix) for k in changed_keys):
        raise SystemExit("ABORT: parameters outside the selected block changed")

    gpu_end.record()
    torch.cuda.synchronize(device)
    gpu_seconds = float(gpu_start.elapsed_time(gpu_end) / 1000.0)
    peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    adapted_checkpoint = output / "adapted_checkpoint.pt"
    tmp_checkpoint = output / ".adapted_checkpoint.pt.tmp"
    # The historical PFPA helper uses bare Hugging Face state keys, while the
    # already-validated FieldDistill P1 evaluator's adapter is rooted at
    # "model.". Prefix keys only; tensor values and update math are unchanged.
    p1_state = {"model." + k: v.detach().cpu() for k, v in state.items()}
    torch.save({"model": p1_state}, tmp_checkpoint)
    os.replace(tmp_checkpoint, adapted_checkpoint)
    checkpoint_sha = sha256(adapted_checkpoint)
    wall_seconds = float(time.perf_counter() - wall_start)

    config = {
        "experiment": {"method": "PFPA", "domain": args.domain, "seed": args.seed, "device": "cuda:0", "visible_device_ordinal": 0, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "output_dir": str(output)},
        "protocol": {"path": str(protocol_path), "sha256": actual_protocol_sha, "runner_sha256": runner_sha256},
        "initialization": {"policy": "SourceOnly checkpoint bound by a validated provenance sidecar", "source_domain_tag": args.source_domain_tag, "checkpoint": str(source_checkpoint), "sha256": source_checkpoint_sha, "provenance_path": str(source_checkpoint_provenance), "provenance_sha256": source_provenance_sha, "provenance_method": validated_source_provenance["method"], "provenance_source_domain": validated_source_provenance["source_domain"], "provenance_source_train_manifest_sha256": validated_source_provenance["source_train_manifest_sha256"], "provenance_split_seed": int(validated_source_provenance["split_seed"])},
        "data": {"source_domain_tag": args.source_domain_tag, "source_train_manifest": str(source_train), "source_train_manifest_sha256": sha256(source_train), "source_val_manifest": str(source_val), "source_val_manifest_sha256": sha256(source_val), "target_train_manifest": str(target_train), "target_train_manifest_sha256": sha256(target_train), "target_val_manifest_historical_diagnostic_only": str(target_val), "target_val_manifest_sha256": sha256(target_val), "target_test_manifest_evaluation_only": str(target_test), "target_test_manifest_sha256": sha256(target_test), "target_label_access_during_adaptation": "target masks/labels are not consumed; image tensors drive target gradients", "source_anchor_images": len(source_ds), "source_anchor_requested_cap": args.image_cap, "source_anchor_effective_images": len(source_ds), "target_calibration_requested_cap": args.image_cap, "target_calibration_effective_images": len(target_ds), "target_sample_ids_sha256": target_ids_sha, "source_sample_ids_sha256": source_ids_sha},
        "update": {"selected_block": selected_block, "module_name": module_name, "trainable_params": trainable_params, "total_model_params": total_params, "trainable_ratio": trainable_params / total_params, "epsilon": EPSILON, "target_objective": "entropy_weight=1.0 + symmetric_JS(horizontal_flip)=0.5", "source_anchor": f"per-batch mean over present-class CE(ignore=255), averaged over {len(source_ds)} seeded {args.source_domain_tag} source-train images", "conflict_projection": "remove target-gradient component along source gradient iff dot(target,source)<0", "optimizer": "none; one normalized projected gradient parameter update"},
        "runtime": {"batch_size": BATCH_SIZE, "workers": WORKERS, "image_size": IMAGE_SIZE, "target_loader_shuffle": False, "source_loader_shuffle": False, "wall_seconds": wall_seconds, "cuda_event_seconds": gpu_seconds, "peak_cuda_memory_allocated_bytes": peak_memory_bytes},
        "artifacts": {"adapted_checkpoint": str(adapted_checkpoint), "adapted_checkpoint_sha256": checkpoint_sha, "checkpoint_format": "FieldDistill P1 strict-load format; historical Hugging Face tensor keys prefixed with model.; tensors unchanged", "selector_csv": str(proxy_path), "selector_csv_sha256": sha256(proxy_path), "selector_json": str(selector_path), "selector_json_sha256": sha256(selector_path)},
    }
    with (output / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    diagnostic = {
        "domain": args.domain,
        "seed": args.seed,
        "selected_block": selected_block,
        "trainable_params": trainable_params,
        "total_model_params": total_params,
        "trainable_ratio": trainable_params / total_params,
        "epsilon": EPSILON,
        "num_target_images": len(target_ds),
        "requested_target_image_cap": args.image_cap,
        "source_anchor_images": len(source_ds),
        "gradient_cosine_before": projection["cos_before"],
        "gradient_cosine_after": projection["cos_after"],
        "conflict_removed_fraction": projection["removed_fraction"],
        "parameter_displacement": measured_displacement,
        "reported_normalized_update_displacement": update["relative_displacement"],
        "target_gradient_norm": float(target_gradient.double().norm().item()),
        "source_gradient_norm": float(source_gradient.double().norm().item()),
        "projected_gradient_norm": float(projected.double().norm().item()),
        "gradient_dot_before": projection["dot_before"],
        "target_objective_mean": float(target_loss),
        "source_anchor_loss_mean": float(source_loss),
        "changed_state_tensor_count": len(changed_keys),
        "changed_state_tensor_keys": changed_keys,
        "all_changes_confined_to_selected_block": True,
        "target_sample_ids_sha256": target_ids_sha,
        "source_sample_ids_sha256": source_ids_sha,
        "protocol_sha256": actual_protocol_sha,
        "source_checkpoint_sha256": source_checkpoint_sha,
        "source_checkpoint_provenance_sha256": source_provenance_sha,
        "adapted_checkpoint_sha256": checkpoint_sha,
        "wall_seconds": wall_seconds,
        "cuda_event_seconds": gpu_seconds,
        "peak_cuda_memory_allocated_bytes": peak_memory_bytes,
    }
    (output / "gradient_diagnostic.json").write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ADAPTATION_PASS", "domain": args.domain, "seed": args.seed, "source_domain_tag": args.source_domain_tag, "selected_block": selected_block, "target_images": len(target_ds), "source_images": len(source_ds), "trainable_params": trainable_params, "trainable_ratio": trainable_params/total_params, "epsilon": EPSILON, "parameter_displacement": measured_displacement, "source_checkpoint_sha256": source_checkpoint_sha, "source_checkpoint_provenance_sha256": source_provenance_sha, "adapted_checkpoint_sha256": checkpoint_sha, "protocol_sha256": actual_protocol_sha, "wall_seconds": wall_seconds, "cuda_event_seconds": gpu_seconds, "peak_cuda_memory_allocated_bytes": peak_memory_bytes}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
