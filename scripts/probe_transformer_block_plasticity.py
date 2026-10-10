from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation


ROOT = Path(os.environ.get("PFPA_DATA_ROOT", Path.cwd())).expanduser().resolve()

IGNORE = 255
NUM_CLASSES = 3


def args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--checkpoint",
        default="models/teachers/phenobench_segformer_b0_best.pt",
    )

    p.add_argument(
        "--pretrained",
        default="models/pretrained/mit-b0",
    )

    p.add_argument(
        "--source-train",
        default="manifests/phenobench_train.csv",
    )

    p.add_argument(
        "--source-val",
        default="manifests/phenobench_val.csv",
    )

    p.add_argument(
        "--target-train",
        default="manifests/cropandweed_sugar_beet_train_unlabeled.csv",
    )

    p.add_argument(
        "--target-val",
        default="manifests/cropandweed_sugar_beet_val.csv",
    )

    p.add_argument(
        "--output",
        default="runs/transformer_function_plasticity_v1",
    )

    p.add_argument(
        "--image-size",
        type=int,
        default=512,
    )

    p.add_argument(
        "--calibration-images",
        type=int,
        default=128,
    )

    p.add_argument(
        "--source-gradient-images",
        type=int,
        default=128,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    p.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    p.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
    )

    p.add_argument(
        "--entropy-weight",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--consistency-weight",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=3407,
    )

    p.add_argument(
        "--blocks",
        nargs="*",
        default=None,
        help="Optional block names, e.g. stage1_block1 stage2_block1",
    )

    return p.parse_args()


def resolve(path):
    path = Path(path)

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def normalize_image(image, size):
    image = TF.resize(
        image,
        [size, size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )

    x = (
        TF.pil_to_tensor(image).float()
        / 255.0
    )

    return TF.normalize(
        x,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


class TrainImages(Dataset):

    def __init__(
        self,
        manifest,
        size,
        maximum,
        seed,
        masks=False,
    ):
        df = pd.read_csv(manifest)

        rng = np.random.default_rng(seed)

        if maximum > 0 and len(df) > maximum:
            ids = rng.choice(
                len(df),
                size=maximum,
                replace=False,
            )
            df = df.iloc[
                np.sort(ids)
            ].copy()

        self.df = df.reset_index(drop=True)
        self.size = size
        self.masks = masks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(
            row["image"]
        ).convert("RGB")

        x = normalize_image(
            image,
            self.size,
        )

        if not self.masks:
            return x

        mask_path = row["mask"]

        if pd.isna(mask_path):
            raise RuntimeError(
                f"Mask is missing for a dataset opened "
                f"with masks=True. "
                f"sample_id={row.get('sample_id', idx)}"
            )

        mask = Image.open(str(mask_path))

        mask = TF.resize(
            mask,
            [self.size, self.size],
            interpolation=InterpolationMode.NEAREST,
        )

        y = torch.from_numpy(
            np.asarray(
                mask,
                dtype=np.int64,
            ).copy()
        ).long()

        return x, y


def load_checkpoint(path):
    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(obj, dict):
        raise RuntimeError(
            f"Invalid checkpoint: {path}"
        )

    for key in [
        "model",
        "state_dict",
        "model_state_dict",
    ]:
        if isinstance(
            obj.get(key),
            dict,
        ):
            return obj[key]

    if obj and all(
        torch.is_tensor(v)
        for v in obj.values()
    ):
        return obj

    raise RuntimeError(
        f"Cannot locate state_dict: {path}"
    )


def best_state_mapping(raw, target_keys):

    candidates = {
        "raw": raw,
    }

    prefixes = [
        "module.",
        "model.",
        "model.model.",
        "module.model.",
        "module.model.model.",
    ]

    for prefix in prefixes:
        converted = {}

        for key, value in raw.items():
            new_key = (
                key[len(prefix):]
                if key.startswith(prefix)
                else key
            )

            converted[new_key] = value

        candidates[
            f"strip:{prefix}"
        ] = converted

    target_keys = set(target_keys)

    best_name = None
    best_state = None
    best_overlap = -1

    for name, state in candidates.items():

        overlap = len(
            set(state.keys())
            & target_keys
        )

        if overlap > best_overlap:
            best_name = name
            best_state = state
            best_overlap = overlap

    print(
        "checkpoint mapping:",
        best_name,
        "overlap:",
        best_overlap,
    )

    return best_state


def build_model(pretrained, checkpoint, device):

    model = (
        SegformerForSemanticSegmentation
        .from_pretrained(
            pretrained,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
    )

    raw = load_checkpoint(
        checkpoint
    )

    state = best_state_mapping(
        raw,
        model.state_dict().keys(),
    )

    missing, unexpected = (
        model.load_state_dict(
            state,
            strict=False,
        )
    )

    critical = [
        key
        for key in missing
        if (
            "decode_head.classifier"
            in key
        )
    ]

    if critical:
        raise RuntimeError(
            f"Classifier not loaded: "
            f"{critical}"
        )

    print(
        "missing:",
        len(missing),
        "unexpected:",
        len(unexpected),
    )

    return model.to(device)


def block_registry(model):

    blocks = []

    encoder_blocks = (
        model.segformer.encoder.block
    )

    for stage_index, stage in enumerate(
        encoder_blocks,
        start=1,
    ):
        for block_index, block in enumerate(
            stage,
            start=1,
        ):

            name = (
                f"stage{stage_index}_"
                f"block{block_index}"
            )

            params = sum(
                p.numel()
                for p in block.parameters()
            )

            blocks.append(
                (
                    name,
                    block,
                    stage_index,
                    block_index,
                    params,
                )
            )

    return blocks


def freeze_strict(model, selected):

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for parameter in selected.parameters():
        parameter.requires_grad_(True)

    # Keep every module in eval mode.
    # Gradients still flow; BN running stats and
    # stochastic dropout/drop-path remain frozen.
    model.eval()


def prediction_entropy(logits):

    p = logits.softmax(dim=1)

    return (
        -(
            p
            * p.clamp_min(1e-8).log()
        ).sum(dim=1)
    ).mean()


def symmetric_js(logits_a, logits_b):

    pa = logits_a.softmax(dim=1)

    pb = logits_b.softmax(dim=1)

    m = 0.5 * (
        pa + pb
    )

    kl_a = (
        pa
        * (
            pa.clamp_min(1e-8).log()
            - m.clamp_min(1e-8).log()
        )
    ).sum(dim=1).mean()

    kl_b = (
        pb
        * (
            pb.clamp_min(1e-8).log()
            - m.clamp_min(1e-8).log()
        )
    ).sum(dim=1).mean()

    return 0.5 * (
        kl_a + kl_b
    )


def target_gradient(
    model,
    block,
    loader,
    device,
    entropy_weight,
    consistency_weight,
):

    freeze_strict(
        model,
        block,
    )

    for p in block.parameters():
        p.grad = None

    loss_sum = 0.0
    batches = 0

    for x in loader:

        x = x.to(
            device,
            non_blocking=True,
        )

        xf = torch.flip(
            x,
            dims=[3],
        )

        logits = model(
            pixel_values=x
        ).logits

        logits_f = model(
            pixel_values=xf
        ).logits

        logits_f = torch.flip(
            logits_f,
            dims=[3],
        )

        entropy = prediction_entropy(
            logits
        )

        consistency = symmetric_js(
            logits,
            logits_f,
        )

        loss = (
            entropy_weight * entropy
            + consistency_weight
            * consistency
        )

        loss.backward()

        loss_sum += float(
            loss.detach().cpu()
        )

        batches += 1

    for p in block.parameters():
        if p.grad is not None:
            p.grad.div_(
                max(batches, 1)
            )

    return (
        loss_sum / max(
            batches,
            1,
        )
    )


def source_gradient(
    model,
    block,
    loader,
    device,
):

    freeze_strict(
        model,
        block,
    )

    for p in block.parameters():
        p.grad = None

    batches = 0
    loss_sum = 0.0

    for x, y in loader:

        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        logits = model(
            pixel_values=x
        ).logits

        y_small = (
            F.interpolate(
                y[:, None].float(),
                size=logits.shape[-2:],
                mode="nearest",
            )
            .squeeze(1)
            .long()
        )

        per_pixel = F.cross_entropy(
            logits,
            y_small,
            reduction="none",
            ignore_index=IGNORE,
        )

        class_losses = []

        for cls in range(
            NUM_CLASSES
        ):
            mask = (
                y_small == cls
            )

            if mask.any():
                class_losses.append(
                    per_pixel[
                        mask
                    ].mean()
                )

        if not class_losses:
            continue

        loss = torch.stack(
            class_losses
        ).mean()

        loss.backward()

        loss_sum += float(
            loss.detach().cpu()
        )

        batches += 1

    for p in block.parameters():

        if p.grad is not None:
            p.grad.div_(
                max(batches, 1)
            )

    return (
        loss_sum / max(
            batches,
            1,
        )
    )


def flatten_grad(block):

    values = []

    for p in block.parameters():

        if p.grad is None:
            values.append(
                torch.zeros_like(
                    p,
                    memory_format=(
                        torch.preserve_format
                    ),
                ).reshape(-1)
            )

        else:
            values.append(
                p.grad.detach().reshape(-1)
            )

    return torch.cat(
        values
    )


def parameter_vector(block):

    return torch.cat(
        [
            p.detach().reshape(-1)
            for p in block.parameters()
        ]
    )


@torch.no_grad()
def apply_normalized_update(
    block,
    gradient,
    epsilon,
):

    params = [
        p
        for p in block.parameters()
    ]

    param_norm_sq = 0.0

    for p in params:
        param_norm_sq += (
            p.detach().double()
            .pow(2).sum().item()
        )

    param_norm = math.sqrt(
        param_norm_sq
    )

    grad_norm = float(
        gradient.double().norm().item()
    )

    if grad_norm <= 1e-12:
        return {
            "parameter_norm":
                param_norm,
            "gradient_norm":
                grad_norm,
            "update_norm":
                0.0,
            "relative_displacement":
                0.0,
        }

    scale = (
        epsilon
        * param_norm
        / grad_norm
    )

    offset = 0

    for p in params:

        n = p.numel()

        g = gradient[
            offset:offset+n
        ].view_as(p)

        p.add_(
            g,
            alpha=-scale,
        )

        offset += n

    update_norm = (
        epsilon
        * param_norm
    )

    return {
        "parameter_norm":
            param_norm,
        "gradient_norm":
            grad_norm,
        "update_norm":
            update_norm,
        "relative_displacement":
            (
                update_norm
                / max(
                    param_norm,
                    1e-12,
                )
            ),
    }


def confusion_update(
    confusion,
    pred,
    gt,
):

    valid = (
        gt != IGNORE
    )

    pred = pred[valid]
    gt = gt[valid]

    encoded = (
        gt * NUM_CLASSES
        + pred
    )

    hist = np.bincount(
        encoded,
        minlength=(
            NUM_CLASSES
            * NUM_CLASSES
        ),
    ).reshape(
        NUM_CLASSES,
        NUM_CLASSES,
    )

    confusion += hist


@torch.inference_mode()
def evaluate(
    model,
    manifest,
    size,
    device,
    desc,
):

    frame = pd.read_csv(
        manifest
    )

    confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64,
    )

    model.eval()

    for row in tqdm(
        frame.itertuples(
            index=False
        ),
        total=len(frame),
        desc=desc,
        leave=False,
    ):

        image = Image.open(
            row.image
        ).convert("RGB")

        gt = np.asarray(
            Image.open(
                row.mask
            ),
            dtype=np.int64,
        )

        x = normalize_image(
            image,
            size,
        )[None].to(device)

        logits = model(
            pixel_values=x
        ).logits

        logits = F.interpolate(
            logits,
            size=gt.shape,
            mode="bilinear",
            align_corners=False,
        )

        pred = (
            logits.argmax(dim=1)[0]
            .cpu().numpy()
            .astype(np.int64)
        )

        confusion_update(
            confusion,
            pred,
            gt,
        )

    tp = np.diag(
        confusion
    ).astype(np.float64)

    fp = (
        confusion.sum(axis=0)
        - tp
    )

    fn = (
        confusion.sum(axis=1)
        - tp
    )

    denom = (
        tp + fp + fn
    )

    iou = np.divide(
        tp,
        denom,
        out=np.zeros_like(tp),
        where=denom > 0,
    )

    return {
        "mIoU":
            float(
                100.0 * iou.mean()
            ),
        "soil_IoU":
            float(
                100.0 * iou[0]
            ),
        "crop_IoU":
            float(
                100.0 * iou[1]
            ),
        "weed_IoU":
            float(
                100.0 * iou[2]
            ),
        "plant_mIoU":
            float(
                50.0
                * (
                    iou[1]
                    + iou[2]
                )
            ),
    }


def main():

    cfg = args()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = resolve(
        cfg.checkpoint
    )

    pretrained = resolve(
        cfg.pretrained
    )

    source_train = resolve(
        cfg.source_train
    )

    source_val = resolve(
        cfg.source_val
    )

    target_train = resolve(
        cfg.target_train
    )

    target_val = resolve(
        cfg.target_val
    )

    output = resolve(
        cfg.output
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------
    # Fixed calibration subsets
    # -----------------------------------------

    target_ds = TrainImages(
        target_train,
        cfg.image_size,
        cfg.calibration_images,
        cfg.seed,
        masks=False,
    )

    source_ds = TrainImages(
        source_train,
        cfg.image_size,
        cfg.source_gradient_images,
        cfg.seed,
        masks=True,
    )

    target_loader = DataLoader(
        target_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
    )

    source_loader = DataLoader(
        source_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
    )

    # -----------------------------------------
    # Baseline
    # -----------------------------------------

    base = build_model(
        pretrained,
        checkpoint,
        device,
    )

    blocks = block_registry(
        base
    )

    if cfg.blocks:
        wanted = set(cfg.blocks)

        blocks = [
            item
            for item in blocks
            if item[0] in wanted
        ]

        missing = (
            wanted
            - {item[0] for item in blocks}
        )

        if missing:
            raise RuntimeError(
                f"Unknown blocks: {sorted(missing)}"
            )

    print(
        "\nBlocks:"
    )

    for item in blocks:
        print(
            item[0],
            "params=",
            item[4],
        )

    print(
        "\nEvaluating baseline..."
    )

    baseline_target = evaluate(
        base,
        target_val,
        cfg.image_size,
        device,
        "baseline target",
    )

    baseline_source = evaluate(
        base,
        source_val,
        cfg.image_size,
        device,
        "baseline source",
    )

    print(
        "baseline target:",
        baseline_target
    )

    print(
        "baseline source:",
        baseline_source
    )

    del base
    torch.cuda.empty_cache()

    rows = []

    # -----------------------------------------
    # Every Transformer block
    # -----------------------------------------

    for (
        name,
        _,
        stage,
        block_index,
        params,
    ) in blocks:

        print(
            "\n"
            + "=" * 110
        )

        print(
            "PROBE:",
            name
        )

        print(
            "=" * 110
        )

        # ----- target gradient -----

        model_t = build_model(
            pretrained,
            checkpoint,
            device,
        )

        selected_t = (
            model_t.segformer
            .encoder.block[
                stage - 1
            ][
                block_index - 1
            ]
        )

        target_loss = target_gradient(
            model_t,
            selected_t,
            target_loader,
            device,
            cfg.entropy_weight,
            cfg.consistency_weight,
        )

        g_target = (
            flatten_grad(
                selected_t
            )
            .detach()
            .cpu()
        )

        del model_t
        torch.cuda.empty_cache()

        # ----- source semantic gradient -----

        model_s = build_model(
            pretrained,
            checkpoint,
            device,
        )

        selected_s = (
            model_s.segformer
            .encoder.block[
                stage - 1
            ][
                block_index - 1
            ]
        )

        source_loss = source_gradient(
            model_s,
            selected_s,
            source_loader,
            device,
        )

        g_source = (
            flatten_grad(
                selected_s
            )
            .detach()
            .cpu()
        )

        cosine = float(
            F.cosine_similarity(
                g_target.double(),
                g_source.double(),
                dim=0,
                eps=1e-12,
            ).item()
        )

        del model_s
        torch.cuda.empty_cache()

        # ----- exact relative-displacement intervention -----

        model = build_model(
            pretrained,
            checkpoint,
            device,
        )

        selected = (
            model.segformer
            .encoder.block[
                stage - 1
            ][
                block_index - 1
            ]
        )

        freeze_strict(
            model,
            selected,
        )

        update_info = (
            apply_normalized_update(
                selected,
                g_target.to(device),
                cfg.epsilon,
            )
        )

        target_result = evaluate(
            model,
            target_val,
            cfg.image_size,
            device,
            f"{name} target",
        )

        source_result = evaluate(
            model,
            source_val,
            cfg.image_size,
            device,
            f"{name} source",
        )

        row = {
            "block":
                name,
            "stage":
                stage,
            "block_index":
                block_index,
            "parameters":
                params,

            "epsilon":
                cfg.epsilon,

            "target_probe_loss":
                target_loss,
            "source_semantic_loss":
                source_loss,

            "gradient_alignment":
                cosine,

            **update_info,

            "base_target_mIoU":
                baseline_target[
                    "mIoU"
                ],
            "target_mIoU":
                target_result[
                    "mIoU"
                ],
            "delta_target_mIoU":
                (
                    target_result[
                        "mIoU"
                    ]
                    - baseline_target[
                        "mIoU"
                    ]
                ),

            "base_source_mIoU":
                baseline_source[
                    "mIoU"
                ],
            "source_mIoU":
                source_result[
                    "mIoU"
                ],
            "delta_source_mIoU":
                (
                    source_result[
                        "mIoU"
                    ]
                    - baseline_source[
                        "mIoU"
                    ]
                ),

            "target_crop_IoU":
                target_result[
                    "crop_IoU"
                ],
            "delta_target_crop":
                (
                    target_result[
                        "crop_IoU"
                    ]
                    - baseline_target[
                        "crop_IoU"
                    ]
                ),

            "target_weed_IoU":
                target_result[
                    "weed_IoU"
                ],
            "delta_target_weed":
                (
                    target_result[
                        "weed_IoU"
                    ]
                    - baseline_target[
                        "weed_IoU"
                    ]
                ),

            "target_plant_mIoU":
                target_result[
                    "plant_mIoU"
                ],
        }

        # Positive = target gain with low source forgetting.
        forgetting = max(
            0.0,
            -row[
                "delta_source_mIoU"
            ],
        )

        row[
            "plasticity_score"
        ] = (
            row[
                "delta_target_mIoU"
            ]
            - forgetting
        )

        rows.append(
            row
        )

        print(
            json.dumps(
                row,
                indent=2,
            )
        )

        pd.DataFrame(
            rows
        ).to_csv(
            output
            / "block_plasticity_partial.csv",
            index=False,
        )

        del model
        torch.cuda.empty_cache()

    frame = pd.DataFrame(
        rows
    )

    frame[
        "plasticity_rank"
    ] = (
        frame[
            "plasticity_score"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    frame[
        "target_gain_rank"
    ] = (
        frame[
            "delta_target_mIoU"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    frame = frame.sort_values(
        "plasticity_rank"
    )

    frame.to_csv(
        output
        / "block_plasticity_matrix.csv",
        index=False,
    )

    metadata = {
        "checkpoint":
            str(checkpoint),
        "target_calibration_images":
            len(target_ds),
        "source_gradient_images":
            len(source_ds),
        "epsilon":
            cfg.epsilon,
        "entropy_weight":
            cfg.entropy_weight,
        "consistency_weight":
            cfg.consistency_weight,
        "seed":
            cfg.seed,
        "baseline_target":
            baseline_target,
        "baseline_source":
            baseline_source,
    }

    (
        output
        / "probe_protocol.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 130
    )

    print(
        "FINAL PLASTICITY MATRIX"
    )

    print(
        "=" * 130
    )

    print(
        frame[
            [
                "block",
                "parameters",
                "gradient_alignment",
                "delta_target_mIoU",
                "delta_source_mIoU",
                "delta_target_crop",
                "delta_target_weed",
                "plasticity_score",
                "plasticity_rank",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:+.4f}",
        )
    )

    print(
        "\nTRANSFORMER_BLOCK_PLASTICITY_PROBE_COMPLETED"
    )


if __name__ == "__main__":
    main()
