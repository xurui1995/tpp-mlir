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
from tpp.dialects import func
from utils import *
from abc import ABC, abstractmethod
import numpy as np
import argparse


class DriverParams(ABC):
    @staticmethod
    @abstractmethod
    def add_args(parser: argparse.ArgumentParser):
        pass

    def __init__(self, params: dict):
        pass


class Driver(ABC):
    def __init__(self, ctx: ir.Context):
        self.ir_module = self.init_module(ctx)

    @abstractmethod
    def init_module(self, ctx: ir.Context) -> ir.Module:
        pass

    @abstractmethod
    def prepare_np_args(self) -> list[np.ndarray]:
        pass

    @abstractmethod
    def prepare_np_res(self) -> list[np.ndarray]:
        pass

    def get_passes(self) -> str:
        return get_default_passes()
