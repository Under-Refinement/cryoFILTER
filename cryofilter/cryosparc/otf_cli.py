"""Mask-card command arguments; keep CLI discovery free of bridge/GPU imports."""

import argparse


def add_parsers(sub):
    parser = sub.add_parser("otf", help="Follow a running Patch Motion Correction job on a shared filesystem.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--micrographs", required=True, help="Patch Motion Correction job, e.g. J12.")
    parser.add_argument("--gpu-devices", required=True, help="Comma-separated GPU IDs reserved for cryoFILTER, e.g. 1,2.")
    parser.add_argument("--num-cpus", type=int, default=8, help="Total CPU budget across all OTF workers; default 8.")
    parser.add_argument("--run-typing", action="store_true", help="Dedicate separate GPU(s) to typing; requires 2+ GPUs.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--typing-checkpoint")
    parser.add_argument("--typing-sample-stride-px", type=int, choices=(16,32,64), default=64)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--inference-profile", choices=("fast","balanced","quality"), default="balanced")
    parser.add_argument("--batch-forward-size", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--max-output-gb", type=float, default=25, help="Maximum cumulative generated data, including restart leftovers.")
    parser.add_argument("--local-run-root", default="cryofilter_runs/otf")
    parser.add_argument("--run-id")
    parser.add_argument("--resume-job", help="Resume an existing OTF card in this project (same source/settings).")
    parser.add_argument("--title", default="cryoFILTER-OTFwMC")
    parser.add_argument("--no-previews", action="store_true")
    filtering = sub.add_parser(
        "filter-otf",
        help="Filter a CryoSPARC particle job using masks from an Inference or OTF card.",
    )
    filtering.add_argument("--project", required=True)
    filtering.add_argument("--workspace", required=True)
    filtering.add_argument("--otf-job", required=True, help="Inference or OTF cryoFILTER job UID.")
    filtering.add_argument("--particles", required=True)
    filtering.add_argument("--particle-exclusion-distance-angstrom", type=float, default=100)
    filtering.add_argument("--missing-masks", choices=("error", "pending"), default="error")
    filtering.add_argument("--output-dir", required=True)
    filtering.add_argument("--title", default="cryoFILTER filtered picks")

    inference_card = sub.add_parser(
        "inference-card",
        help="Run batch inference while publishing a reusable CryoSPARC mask card.",
    )
    inference_card.add_argument("--project", required=True)
    inference_card.add_argument("--workspace", required=True)
    inference_card.add_argument("--local-run-dir", required=True)
    inference_card.add_argument("--output-dir", required=True)
    inference_card.add_argument("--poll-seconds", type=float, default=2.0)
    inference_card.add_argument(
        "--run-typing",
        action="store_true",
        help="Run publication contamination typing after segmentation.",
    )
    inference_card.add_argument("--typing-workers", type=int)
    inference_card.add_argument(
        "--typing-sample-stride-px",
        type=int,
        choices=(16, 32, 64),
        default=64,
    )
    inference_card.add_argument(
        "infer_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are forwarded to cryofilter infer.",
    )
