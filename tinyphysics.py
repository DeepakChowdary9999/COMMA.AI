import argparse
import importlib
import numpy as np
import onnxruntime as ort
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import signal
import urllib.request
import zipfile
import io

from io import BytesIO
from collections import namedtuple
from functools import partial
from hashlib import md5
from pathlib import Path
from typing import Union
from tqdm.contrib.concurrent import process_map

from controllers import BaseController

sns.set_theme()
signal.signal(signal.SIGINT, signal.SIG_DFL)  # Enable Ctrl-C on plot windows

# Simulation parameters
ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0

FUTURE_PLAN_STEPS = FPS * 5  # 5 secs

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])

# Dataset config
DATASET_URL = (
    "https://huggingface.co/datasets/commaai/commaSteeringControl/"
    "resolve/main/data/SYNTHETIC_V0.zip"
)
DATASET_PATH = Path(__file__).resolve().parent / "data"


def plot_platform_distribution(data_source: str):
    """
    Plot segment counts per platform.
    If data_source is a URL, streams the ZIP and inspects its CSV entries.
    Otherwise treats it as a local directory of CSVs.
    """
    # Gather list of CSV filenames
    if data_source.startswith(("http://", "https://")):
        resp = urllib.request.urlopen(data_source)
        if resp.getcode() != 200:
            raise RuntimeError(f"Failed to fetch {data_source}: HTTP {resp.getcode()}")
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        files = [n for n in z.namelist() if n.endswith(".csv")]
    else:
        files = [f for f in os.listdir(data_source) if f.endswith('.csv')]

    # Extract platform names (strip off trailing _<digits>.csv)
    platforms = [
        re.sub(r'_\d+\.csv$', '', os.path.basename(f)).replace('_', ' ')
        for f in files
    ]

    # Count and plot
    df = pd.DataFrame({'platform': platforms})
    counts = df['platform'].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(16, 6))
    counts.plot(kind='bar')
    plt.title('CommaSteeringControl Dataset')
    plt.xlabel('Platform')
    plt.ylabel('Segments')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


class LataccelTokenizer:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

    def encode(self, value):
        value = self.clip(value)
        return np.digitize(value, self.bins, right=True)

    def decode(self, token):
        return self.bins[token]

    def clip(self, value):
        return np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])


class TinyPhysicsModel:
    def __init__(self, model_path: str, debug: bool) -> None:
        self.tokenizer = LataccelTokenizer()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        provider = 'CPUExecutionProvider'
        with open(model_path, 'rb') as f:
            self.ort_session = ort.InferenceSession(f.read(), options, [provider])

    def softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def predict(self, input_data: dict, temperature=1.) -> int:
        res = self.ort_session.run(None, input_data)[0]
        probs = self.softmax(res / temperature, axis=-1)
        return np.random.choice(probs.shape[-1], p=probs[0, -1])

    def get_current_lataccel(self, sim_states, actions, past_preds):
        tokenized = self.tokenizer.encode(past_preds)
        raw = [list(s) for s in sim_states]
        states = np.column_stack([actions, raw])
        input_data = {
            'states': np.expand_dims(states, 0).astype(np.float32),
            'tokens': np.expand_dims(tokenized, 0).astype(np.int64)
        }
        return self.tokenizer.decode(self.predict(input_data, temperature=0.8))


class TinyPhysicsSimulator:
    def __init__(self, model, data_source: Union[str, io.IOBase], controller: BaseController, debug: bool=False):
        self.sim_model = model
        self.data = self._load_data(data_source)
        self.controller = controller
        self.debug = debug
        self.reset()

    def _load_data(self, path_or_buf):
        df = pd.read_csv(path_or_buf)
        return pd.DataFrame({
            'roll_lataccel': np.sin(df['roll']) * ACC_G,
            'v_ego': df['vEgo'],
            'a_ego': df['aEgo'],
            'target_lataccel': df['targetLateralAcceleration'],
            'steer_command': -df['steerCommand']
        })

    def reset(self):
        self.step_idx = CONTEXT_LENGTH
        init = [self.get_state_target_futureplan(i) for i in range(self.step_idx)]
        self.state_history = [s for s, _, _ in init]
        self.action_history = self.data['steer_command'][:self.step_idx].tolist()
        self.current_lataccel_history = [lat for _, lat, _ in init]
        self.target_lataccel_history = [lat for _, lat, _ in init]
        self.current_lataccel = self.current_lataccel_history[-1]
        seed = int(md5(str(self.data).encode()).hexdigest(), 16) % 10**6
        np.random.seed(seed)

    def get_state_target_futureplan(self, idx):
        row = self.data.iloc[idx]
        future = self.data.iloc[idx+1:idx+1+FUTURE_PLAN_STEPS]
        return (
            State(row['roll_lataccel'], row['v_ego'], row['a_ego']),
            row['target_lataccel'],
            FuturePlan(
                lataccel=future['target_lataccel'].tolist(),
                roll_lataccel=future['roll_lataccel'].tolist(),
                v_ego=future['v_ego'].tolist(),
                a_ego=future['a_ego'].tolist()
            )
        )

    def control_step(self):
        steer = self.controller.update(
            self.target_lataccel_history[self.step_idx],
            self.current_lataccel,
            self.state_history[self.step_idx],
            self.futureplan
        )
        action = np.clip(steer, STEER_RANGE[0], STEER_RANGE[1])
        self.action_history.append(action)

    def sim_step(self):
        pred = self.sim_model.get_current_lataccel(
            self.state_history[-CONTEXT_LENGTH:],
            self.action_history[-CONTEXT_LENGTH:],
            self.current_lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = np.clip(pred,
                       self.current_lataccel - MAX_ACC_DELTA,
                       self.current_lataccel + MAX_ACC_DELTA)
        self.current_lataccel = (
            pred if self.step_idx >= CONTROL_START_IDX
            else self.target_lataccel_history[self.step_idx]
        )
        self.current_lataccel_history.append(self.current_lataccel)

    def step(self):
        state, target, self.futureplan = self.get_state_target_futureplan(self.step_idx)
        self.state_history.append(state)
        self.target_lataccel_history.append(target)
        self.control_step()
        self.sim_step()
        self.step_idx += 1

    def compute_cost(self):
        targ = np.array(self.target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
        pred = np.array(self.current_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
        lat_cost = np.mean((targ - pred)**2) * 100
        jerk_cost = np.mean((np.diff(pred)/DEL_T)**2) * 100
        return {
            'lataccel_cost': lat_cost,
            'jerk_cost': jerk_cost,
            'total_cost': lat_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
        }

    def rollout(self):
        if self.debug:
            plt.ion()
            fig, axs = plt.subplots(2, figsize=(10, 6))
        while self.step_idx < len(self.data):
            self.step()
        if self.debug:
            plt.ioff()
            plt.show()
        return self.compute_cost()


def get_available_controllers():
    return [f.stem for f in Path('controllers').iterdir()
            if f.is_file() and f.suffix == '.py'
            and f.stem not in ('__init__', 'base_controller')]


def run_rollout(data_source, controller_type, model_path, debug=False):
    """
    data_source can be either:
      - a filesystem path (str or Path)
      - a file-like object (e.g. from zipfile.ZipFile.open)
    """
    model = TinyPhysicsModel(model_path, debug)
    module = importlib.import_module(f'controllers.{controller_type}')
    controller = module.Controller(params=None)
    sim = TinyPhysicsSimulator(model, data_source, controller, debug)
    return sim.rollout(), None, None


def download_dataset():
    DATASET_PATH.mkdir(exist_ok=True)
    with urllib.request.urlopen(DATASET_URL) as resp:
        with zipfile.ZipFile(BytesIO(resp.read())) as z:
            z.extractall(DATASET_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True,
        help='Local CSV file / directory, or HTTP URL to a ZIP of CSVs')
    parser.add_argument(
        '--plot_platforms', action='store_true',
        help='Plot segment count per platform and exit')
    parser.add_argument(
        '--model_path', type=str,
        help='ONNX model for TinyPhysics (required unless plotting)')
    parser.add_argument(
        '--num_segs', type=int, default=100,
        help='Number of segments to run in batch mode')
    parser.add_argument(
        '--controller', type=str, default='pid_controller',
        choices=get_available_controllers(),
        help='Controller module to use')
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug plots during rollout')
    args = parser.parse_args()

    # Plot-only mode
    if args.plot_platforms:
        plot_platform_distribution(args.data_path)
        exit(0)

    # Simulation mode requires model
    if not args.model_path:
        parser.error('--model_path is required unless --plot_platforms is set')

    # HTTP-ZIP streaming branch
    if args.data_path.startswith(("http://", "https://")):
        resp = urllib.request.urlopen(args.data_path)
        if resp.getcode() != 200:
            raise RuntimeError(f"Failed to fetch {args.data_path}: HTTP {resp.getcode()}")
        z = zipfile.ZipFile(io.BytesIO(resp.read()))

        csv_names = [n for n in z.namelist() if n.endswith(".csv")]
        to_run = csv_names[:args.num_segs]

        costs = []
        for name in to_run:
            with z.open(name) as f:
                cost, _, _ = run_rollout(f, args.controller, args.model_path, debug=args.debug)
            costs.append(cost)

        df = pd.DataFrame(costs)
        print(f"Average lataccel_cost: {df['lataccel_cost'].mean():.4f}, "
              f"jerk_cost:        {df['jerk_cost'].mean():.4f}, "
              f"total_cost:       {df['total_cost'].mean():.4f}")
        for col in ['lataccel_cost', 'jerk_cost', 'total_cost']:
            plt.hist(df[col], bins=50, alpha=0.5, label=col)
        plt.legend()
        plt.show()

    else:
        # on-disk path
        data_path = Path(args.data_path)
        if data_path.is_file():
            cost, _, _ = run_rollout(str(data_path), args.controller, args.model_path, debug=args.debug)
            print(f"Average lataccel_cost: {cost['lataccel_cost']:.4f}, "
                  f"jerk_cost: {cost['jerk_cost']:.4f}, "
                  f"total_cost: {cost['total_cost']:.4f}")
        else:
            run_partial = partial(run_rollout, controller_type=args.controller,
                                  model_path=args.model_path, debug=args.debug)
            files = sorted(data_path.iterdir())[:args.num_segs]
            results = process_map(run_partial, files, max_workers=16, chunksize=10)
            costs = [r[0] for r in results]
            df = pd.DataFrame(costs)
            print(f"Average lataccel_cost: {df['lataccel_cost'].mean():.4f}, "
                  f"jerk_cost: {df['jerk_cost'].mean():.4f}, "
                  f"total_cost: {df['total_cost'].mean():.4f}")
            for col in ['lataccel_cost', 'jerk_cost', 'total_cost']:
                plt.hist(df[col], bins=50, alpha=0.5, label=col)
            plt.legend()
            plt.show()
