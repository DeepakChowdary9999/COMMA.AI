import argparse
import base64
import io
import os
import zipfile
import urllib.request
from io import BytesIO
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from tinyphysics import CONTROL_START_IDX, get_available_controllers, run_rollout

sns.set_theme()
SAMPLE_ROLLOUTS = 5

COLORS = {
    'test': '#c0392b',
    'baseline': '#2980b9'
}


def img2base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    return base64.b64encode(buf.getbuffer()).decode("ascii")


def create_report(test, baseline, sample_rollouts, costs, num_segs):
    # ... your existing create_report implementation unchanged ...
    pass


def download_and_extract_zip(url: str, extract_to: Path):
    resp = urllib.request.urlopen(url)
    if resp.getcode() != 200:
        raise RuntimeError(f"Failed to fetch {url}: HTTP {resp.getcode()}")
    z = zipfile.ZipFile(io.BytesIO(resp.read()))
    extract_to.mkdir(parents=True, exist_ok=True)
    z.extractall(extract_to)


if __name__ == "__main__":
    available_controllers = get_available_controllers()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",     type=str, required=True)
    parser.add_argument("--data_path",      type=str, required=True,
                        help="Local dir of CSVs or HTTP URL to ZIP of CSVs")
    parser.add_argument("--num_segs",       type=int, default=100)
    parser.add_argument("--test_controller",    default='pid', choices=available_controllers)
    parser.add_argument("--baseline_controller", default='pid', choices=available_controllers)
    args = parser.parse_args()

    # If data_path is URL, download & extract to ./tmp_data
    if args.data_path.startswith(("http://", "https://")):
        data_dir = Path("tmp_data")
        download_and_extract_zip(args.data_path, data_dir)
    else:
        data_dir = Path(args.data_path)
        assert data_dir.is_dir(), "data_path should be a directory"

    # Recursively find all CSVs (handles nested folders in the ZIP)
    csv_files = sorted(data_dir.rglob("*.csv"))[: args.num_segs]

    # Sample rollouts
    costs = []
    sample_rollouts = []
    for entry in tqdm(csv_files[:SAMPLE_ROLLOUTS], desc="Sample rollouts"):
        # test controller
        test_cost, test_target, test_current = run_rollout(
            str(entry), args.test_controller, args.model_path, debug=False
        )
        # baseline controller
        base_cost, base_target, base_current = run_rollout(
            str(entry), args.baseline_controller, args.model_path, debug=False
        )

        sample_rollouts.append({
            'seg': entry.stem,
            'desired_lataccel': test_target,
            'test_controller_lataccel': test_current,
            'baseline_controller_lataccel': base_current
        })
        costs.append({'controller': 'test',     **test_cost})
        costs.append({'controller': 'baseline', **base_cost})

    # Batch rollouts
    for controller_cat, controller_type in [('baseline', args.baseline_controller),
                                            ('test', args.test_controller)]:
        desc = f"Batch rollouts ({controller_cat})"
        fn = partial(run_rollout,
                     controller_type=controller_type,
                     model_path=args.model_path,
                     debug=False)
        results = process_map(fn,
                              csv_files[SAMPLE_ROLLOUTS:],
                              max_workers=16,
                              chunksize=10,
                              desc=desc)
        costs += [{'controller': controller_cat, **r[0]} for r in results]

    create_report(
        args.test_controller,
        args.baseline_controller,
        sample_rollouts,
        costs,
        len(csv_files)
    )
