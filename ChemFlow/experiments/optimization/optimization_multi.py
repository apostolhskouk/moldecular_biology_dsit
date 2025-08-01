import torch
import numpy as np
import pandas as pd
from accelerate.utils import set_seed

from absl import logging
import os
from collections import defaultdict
from pandarallel import pandarallel
from tap import Tap
from tqdm import trange
from typing import Literal
import pickle
import random

from rdkit import Chem
from pathlib import Path


from ChemFlow.experiments.utils.traversal_step import Traversal
from ChemFlow.src.utils.scores import *


class Args(Tap):
    prop1: str = "qed"  # property to optimize
    prop2: str = "sa"  # property to optimize
    n: int = 800  # number of molecules to generate
    steps: int = 1000  # number of optimization steps
    method: Literal[
        "random",
        "random_1d",
        "fp",
        "limo",
        #"chemspace",
        "wave_sup",
        "wave_unsup",
        "hj_sup",
        "hj_unsup",
    ] = "random"  # optimization method
    step_size: float = 0.1  # step size
    relative: bool = False  # relative step size
    data_name: str = "zmc"  # data name

    def process_args(self):
        self.model_name = self.prop1 + "_" + self.prop2 + "_" + self.method
        self.model_name += f"_{self.step_size}"
        self.model_name += "_relative" if self.relative else "_absolute"


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    pandarallel.initialize(nb_workers=os.cpu_count(), progress_bar=False, verbose=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(42)

    args = Args().parse_args()
    
    methods = [
        "random",
        "random_1d",
        "fp",
        "limo",
        "wave_sup",
        "wave_unsup",
        "hj_sup",
        "hj_unsup",
    ]
    
    
    traversal1 = Traversal(
        method=args.method,
        prop=args.prop1,
        data_name=args.data_name,
        step_size=args.step_size,
        relative=args.relative,
        minimize=args.prop1 in MINIMIZE_PROPS,
        device=device,
    )
    traversal2 = Traversal(
        method=args.method,
        prop=args.prop2,
        data_name=args.data_name,
        step_size=args.step_size,
        relative=args.relative,
        minimize=args.prop2 in MINIMIZE_PROPS,
        device=device,
    )
    for method in methods:
        # update args for this method
        args.method = method
        args.process_args()
        print(f"\n\n===== Running method: {args.method} ({args.model_name}) =====")

        # build Traversal objects
        traversal1 = Traversal(
            method=args.method,
            prop=args.prop1,
            data_name=args.data_name,
            step_size=args.step_size,
            relative=args.relative,
            minimize=args.prop1 in MINIMIZE_PROPS,
            device=device,
        )
        traversal2 = Traversal(
            method=args.method,
            prop=args.prop2,
            data_name=args.data_name,
            step_size=args.step_size,
            relative=args.relative,
            minimize=args.prop2 in MINIMIZE_PROPS,
            device=device,
        )

        # load and scale properties
        df = pd.read_csv("ChemFlow/data/interim/props/zinc250k.csv")[
            ["smiles", args.prop1, args.prop2]
        ]
        df[args.prop1] = (
            (df[args.prop1] - df[args.prop1].min()) /
            (df[args.prop1].max() - df[args.prop1].min()) * 50
        )
        if args.prop1 in MINIMIZE_PROPS:
            df[args.prop1] = 50 - df[args.prop1]
        df[args.prop2] = (
            (df[args.prop2] - df[args.prop2].min()) /
            (df[args.prop2].max() - df[args.prop2].min()) * 50
        )
        if args.prop2 in MINIMIZE_PROPS:
            df[args.prop2] = 50 - df[args.prop2]

        df["score"] = df[args.prop1] + df[args.prop2]
        df = df.sort_values(["score", args.prop1, args.prop2], ascending=True)

        smiles_raw = df["smiles"].values[: args.n]
        smiles = [s.strip() for s in smiles_raw]
        x = traversal1.dm.encode(smiles).to(device)
        z, *_ = traversal1.vae.encode(x)
        del smiles, x, df

        optimizer = None
        if args.method == "limo":
            z = z.clone().detach()
            z.requires_grad = True
            optimizer = torch.optim.Adam([z], lr=args.step_size)

        print(f"prop1: {args.prop1}")
        print(f"prop2: {args.prop2}")
        results = []

        for t in trange(args.steps):
            for traversal in (traversal1, traversal2):
                u_z = traversal.step(z, t)
                if args.method == "limo":
                    traversal.step(z, t + 1, optimizer=optimizer)
                else:
                    z += u_z
            smiles = traversal.dm.decode(traversal.vae.decode(z))
            results.extend(
                {"idx": i, "t": t, "smiles": s}
                for i, s in enumerate(smiles)
            )

        df = pd.DataFrame(results)
        df_unique = df[["smiles"]].drop_duplicates("smiles")

        def func(_x: pd.Series):
            _x[args.prop1] = PROP_FN[args.prop1](_x["smiles"])
            _x[args.prop2] = PROP_FN[args.prop2](_x["smiles"])
            return _x

        print("Calculating properties, this may take a while...")
        df_unique = df_unique.parallel_apply(func, axis=1)
        df = df.merge(df_unique, on="smiles", how="left")

        # save per-method results
        output_path = Path("ChemFlow/data/interim/optimization")
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = output_path / f"{args.model_name}.csv"
        df.to_csv(out_file, index=False)
        print(f"Results saved to {out_file}")