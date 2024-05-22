# ===============================================================================
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
from tpp import ir
from tpp.passmanager import *
import argparse
import utils
from driver import *
from load_mlir import *
from mlp import *
from timeit import timeit, repeat
import os
import numpy as np
from bench import *
from tuner import *

from tpp.dialects import check,perf,xsmm


def get_driver(args, ctx):
    clz = {"mlp": MLP, "load_mlir": LoadMLIR}[args.driver]
    return clz(ctx, vars(args))


def do_bench(args):
    with ir.Context() as ctx, ir.Location.unknown():
        utils.register_gc_dialects()
        check.register_dialect()
        perf.register_dialect()
        xsmm.register_dialect()
      
        
        
        driver = get_driver(args, ctx)
        # todo
        if isinstance(driver, MLP):
            print("can not bench onednn graph dialect for now")
            return

        if args.print_ir:
            ctx.enable_multithreading(False)
        np_res = driver.prepare_np_res()
        np_args = driver.prepare_np_args()

        # todo need data filling
        # for temporary test
        np.ndarray.fill(np_args[0], 1.0)
        np.ndarray.fill(np_args[1], 1.0)

        mlir_args = np_res_to_mlir_res(np_res) + np_args_to_mlir_args(np_args)

        print("===========bench func name: ", driver.main_entry, "===========")
        python_bench(
            ctx,
            driver.ir_module,
            driver.main_entry,
            driver.get_passes(),
            mlir_args,
            args.print_ir,
            args.repeat,
            args.warm_up,
        )
        print("===========bench over===========")
        # c = rt.ranked_memref_to_numpy(mlir_args[0][0])
        # print(c)


def check_args_and_env(args):
    c_runner_utils = os.getenv("MLIR_C_RUNNER_UTILS", "")
    runner_utils = os.getenv("MLIR_RUNNER_UTILS", "")
    assert os.path.exists(
        c_runner_utils
    ), f"{c_runner_utils} not found, please set a valid MLIR_C_RUNNER_UTILS"
    assert os.path.exists(
        runner_utils
    ), f"{runner_utils} not found, please set a valid MLIR_RUNNER_UTILS"


def add_driver_args(parser):
    driver = parser.parse_known_args()[0].driver
    driver_to_params: dict[str, DriverParams] = {
        "load_mlir": LoadMLIRParams,
        "mlp": MLPParams,
    }
    driver_to_params[driver].add_args(parser)


def do_tune(args):
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        utils.register_gc_dialects()
        driver = get_driver(args, ctx)
        for op in driver.ir_module.operation.regions[0].blocks[0]:
            for j in op.regions[0].blocks[0]:
                print(j)
        # todo (data filling)
        np_res = driver.prepare_np_res()
        np_args = driver.prepare_np_args()

        space = TuningSpace(driver.ir_module)
        if args.search_alg == "grid":
            tuner = GridTuner(
                fake_bench,
                space,
                args.tuning_batch,
                args.early_stop,
                args.checkpoint_path,
            )
        else:
            tuner = GATuner(
                fake_bench,
                space,
                args.tuning_batch,
                args.early_stop,
                args.checkpoint_path,
            )
        tuner.run(args.tuning_times, None, -1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["bench", "tune"], default="bench")
    parser.add_argument(
        "--driver", type=str, choices=["load_mlir", "mlp"], required=True
    )
    add_driver_args(parser)
    if parser.parse_known_args()[0].type == "bench":
        parser.add_argument("-p", "--print_ir", action="store_true", help="")
        parser.add_argument("--warm_up", type=int, default=20, help="")
        parser.add_argument("--repeat", type=int, default=100, help="")
        args = parser.parse_args()
        check_args_and_env(args)
        do_bench(args)
    else:
        parser.add_argument(
            "--search_alg", type=str, choices=["grid", "ga"], default="ga"
        )
        parser.add_argument("--tuning_batch", type=int, default=50)
        parser.add_argument("--early_stop", type=int, default=-1)
        parser.add_argument("--tuning_times", type=int, default=100)
        parser.add_argument("--space_percent", type=float, default=1.0)
        parser.add_argument("--checkpoint_path", type=str, default="")
        args = parser.parse_args()
        check_args_and_env(args)
        do_tune(args)
