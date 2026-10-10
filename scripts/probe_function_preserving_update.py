from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from probe_transformer_block_plasticity import (
    ROOT,
    TrainImages,
    build_model,
    target_gradient,
    source_gradient,
    flatten_grad,
    apply_normalized_update,
    freeze_strict,
    evaluate,
)


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--checkpoint",
        default=(
            "models/teachers/"
            "phenobench_segformer_b0_best.pt"
        ),
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
        required=True,
    )

    p.add_argument(
        "--target-val",
        required=True,
    )

    p.add_argument(
        "--proxy-matrix",
        required=True,
    )

    p.add_argument(
        "--output",
        required=True,
    )

    p.add_argument(
        "--images",
        type=int,
        default=128,
    )

    p.add_argument(
        "--image-size",
        type=int,
        default=512,
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
        "--seed",
        type=int,
        default=3407,
    )

    return p.parse_args()


def resolve(x):
    p = Path(x)

    if not p.is_absolute():
        p = ROOT / p

    return p.resolve()


def parse_block(name):

    # stage1_block1
    left, right = name.split("_")

    stage = int(
        left.replace(
            "stage",
            "",
        )
    )

    block = int(
        right.replace(
            "block",
            "",
        )
    )

    return stage - 1, block - 1


def selected_block(
    model,
    stage,
    block,
):
    return (
        model.segformer
        .encoder.block[
            stage
        ][
            block
        ]
    )


def project_conflict(
    target_gradient,
    anchor_gradient,
):

    gt = (
        target_gradient
        .double()
    )

    ga = (
        anchor_gradient
        .double()
    )

    dot = torch.dot(
        gt,
        ga,
    )

    denom = torch.dot(
        ga,
        ga,
    ).clamp_min(
        1e-20
    )

    if dot < 0:

        projected = (
            gt
            - (
                dot / denom
            ) * ga
        )

    else:
        projected = gt.clone()

    cos_before = float(
        F.cosine_similarity(
            gt,
            ga,
            dim=0,
            eps=1e-12,
        ).item()
    )

    cos_after = float(
        F.cosine_similarity(
            projected,
            ga,
            dim=0,
            eps=1e-12,
        ).item()
    )

    removed_fraction = float(
        (
            gt - projected
        ).norm().item()
        / max(
            gt.norm().item(),
            1e-12,
        )
    )

    return (
        projected.float(),
        {
            "dot_before":
                float(dot.item()),

            "cos_before":
                cos_before,

            "cos_after":
                cos_after,

            "removed_fraction":
                removed_fraction,
        },
    )


def main():

    cfg = parse_args()

    torch.manual_seed(
        cfg.seed
    )

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

    proxy_matrix = resolve(
        cfg.proxy_matrix
    )

    output = resolve(
        cfg.output
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------
    # Label-free block selection
    # ----------------------------------

    proxy = pd.read_csv(
        proxy_matrix
    )

    best = proxy.loc[
        proxy[
            "target_potential"
        ].idxmax()
    ]

    block_name = str(
        best["block"]
    )

    stage, block_index = (
        parse_block(
            block_name
        )
    )

    print(
        "SELECTED BLOCK:",
        block_name
    )

    print(
        "TARGET POTENTIAL:",
        best[
            "target_potential"
        ]
    )

    # ----------------------------------
    # Calibration loaders
    # ----------------------------------

    target_ds = TrainImages(
        target_train,
        cfg.image_size,
        cfg.images,
        cfg.seed,
        masks=False,
    )

    source_ds = TrainImages(
        source_train,
        cfg.image_size,
        cfg.images,
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

    # ----------------------------------
    # Baseline
    # ----------------------------------

    baseline_model = (
        build_model(
            pretrained,
            checkpoint,
            device,
        )
    )

    baseline_target = evaluate(
        baseline_model,
        target_val,
        cfg.image_size,
        device,
        "baseline target",
    )

    baseline_source = evaluate(
        baseline_model,
        source_val,
        cfg.image_size,
        device,
        "baseline source",
    )

    del baseline_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        "BASE TARGET:",
        baseline_target
    )

    print(
        "BASE SOURCE:",
        baseline_source
    )

    # ----------------------------------
    # Target functional/plastic gradient
    # ----------------------------------

    mt = build_model(
        pretrained,
        checkpoint,
        device,
    )

    bt = selected_block(
        mt,
        stage,
        block_index,
    )

    target_gradient(
        mt,
        bt,
        target_loader,
        device,
        entropy_weight=1.0,
        consistency_weight=0.5,
    )

    gt = (
        flatten_grad(bt)
        .detach()
        .cpu()
    )

    del mt

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------------------------
    # Recognition functional anchor
    # ----------------------------------

    ms = build_model(
        pretrained,
        checkpoint,
        device,
    )

    bs = selected_block(
        ms,
        stage,
        block_index,
    )

    source_gradient(
        ms,
        bs,
        source_loader,
        device,
    )

    gr = (
        flatten_grad(bs)
        .detach()
        .cpu()
    )

    del ms

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    projected, projection_info = (
        project_conflict(
            gt,
            gr,
        )
    )

    print(
        "PROJECTION:",
        json.dumps(
            projection_info,
            indent=2,
        )
    )

    modes = {
        "raw_plasticity":
            gt,

        "recognition_projected":
            projected,
    }

    rows = []

    for mode, direction in (
        modes.items()
    ):

        print(
            "\n"
            + "=" * 100
        )

        print(
            "MODE:",
            mode
        )

        print(
            "=" * 100
        )

        model = build_model(
            pretrained,
            checkpoint,
            device,
        )

        block_module = (
            selected_block(
                model,
                stage,
                block_index,
            )
        )

        freeze_strict(
            model,
            block_module,
        )

        update_info = (
            apply_normalized_update(
                block_module,
                direction.to(device),
                cfg.epsilon,
            )
        )

        target_result = evaluate(
            model,
            target_val,
            cfg.image_size,
            device,
            f"{mode} target",
        )

        source_result = evaluate(
            model,
            source_val,
            cfg.image_size,
            device,
            f"{mode} source",
        )

        delta_target = (
            target_result["mIoU"]
            - baseline_target["mIoU"]
        )

        delta_source = (
            source_result["mIoU"]
            - baseline_source["mIoU"]
        )

        rows.append({
            "mode":
                mode,

            "block":
                block_name,

            "epsilon":
                cfg.epsilon,

            "target_mIoU":
                target_result["mIoU"],

            "delta_target_mIoU":
                delta_target,

            "source_mIoU":
                source_result["mIoU"],

            "delta_source_mIoU":
                delta_source,

            "crop_IoU":
                target_result[
                    "crop_IoU"
                ],

            "weed_IoU":
                target_result[
                    "weed_IoU"
                ],

            "plant_mIoU":
                target_result[
                    "plant_mIoU"
                ],

            "relative_displacement":
                update_info[
                    "relative_displacement"
                ],
        })

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(
        rows
    )

    raw_target = float(
        df.loc[
            df["mode"]
            == "raw_plasticity",
            "delta_target_mIoU",
        ].iloc[0]
    )

    projected_target = float(
        df.loc[
            df["mode"]
            == "recognition_projected",
            "delta_target_mIoU",
        ].iloc[0]
    )

    raw_source = float(
        df.loc[
            df["mode"]
            == "raw_plasticity",
            "delta_source_mIoU",
        ].iloc[0]
    )

    projected_source = float(
        df.loc[
            df["mode"]
            == "recognition_projected",
            "delta_source_mIoU",
        ].iloc[0]
    )

    target_retention = (
        projected_target
        / max(
            raw_target,
            1e-12,
        )
    )

    forgetting_raw = max(
        0.0,
        -raw_source,
    )

    forgetting_projected = max(
        0.0,
        -projected_source,
    )

    df[
        "target_gain_retention"
    ] = target_retention

    df.to_csv(
        output
        / "function_preserving_results.csv",
        index=False,
    )

    summary = {
        "selected_block":
            block_name,

        "target_potential":
            float(
                best[
                    "target_potential"
                ]
            ),

        **projection_info,

        "raw_target_gain":
            raw_target,

        "projected_target_gain":
            projected_target,

        "target_gain_retention":
            target_retention,

        "raw_source_forgetting":
            forgetting_raw,

        "projected_source_forgetting":
            forgetting_projected,
    }

    (
        output
        / "function_preserving_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "FUNCTION-PRESERVING UPDATE"
    )

    print(
        "=" * 110
    )

    print(
        df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:+.5f}",
        )
    )

    print(
        "\nSUMMARY:"
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\nFUNCTION_PRESERVING_GATE_COMPLETED"
    )


if __name__ == "__main__":
    main()
