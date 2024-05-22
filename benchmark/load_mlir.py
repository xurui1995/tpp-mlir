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
from driver import *
from tpp import ir
from tpp.dialects import func
from utils import *


class LoadMLIRParams(DriverParams):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument("--path", type=str, required=True)
        parser.add_argument("--entry", type=str, default="main_entry")

    def __init__(self, params: dict):
        self.path = params["path"]
        self.main_entry = params["entry"]


class LoadMLIR(Driver):
    def __init__(self, ctx: ir.Context, params: dict):
        self.params = LoadMLIRParams(params)
        self.main_entry = self.params.main_entry
        super().__init__(ctx)

    def _get_mlir(self):
        with open(self.params.path, "r") as file:
            content = file.read()
        return content

    def init_module(self, ctx: ir.Context) -> ir.Module:
        module = ir.Module.parse(self._get_mlir(), ctx)
        return module

    def prepare_np_args(self) -> list[np.ndarray]:
        bench_func = get_kernel_func_from_module(self.ir_module, self.main_entry)
        np_args = []
        for arg in bench_func.arguments:
            np_args.append(make_tensor(arg.type))
        return np_args

    def prepare_np_res(self) -> list[np.ndarray]:
        bench_func = get_kernel_func_from_module(self.ir_module, self.main_entry)
        np_res = []
        for res in bench_func.type.results:
            np_res.append(make_tensor(res))
        return np_res
