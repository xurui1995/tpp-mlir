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
class Config:
    def __init__(self):
        self.field_candidates = {}
        self.field_constraints = {}
        self.init_candidates()
        self.init_constraints()

    def init_candidates(self):
        pass

    def init_constraints(self):
        pass


class MatMulConfig(Config):
    def __init__(self, M_block: int = 32, K_block: int = 32, N_block: int = 32):
        super().__init__()
        self.M_block = M_block
        self.K_block = K_block
        self.N_block = N_block

    def init_candidates(self):
        self.field_candidates["M_block"] = [16, 32]
        self.field_candidates["K_block"] = [16, 32, 64]
        self.field_candidates["N_block"] = [16]

    def init_constraints(self):
        self.field_constraints["M_block"] = None
        self.field_constraints["K_block"] = (
            lambda MatMulConfig, K_block: MatMulConfig.M_block <= K_block
        )
        self.field_constraints["N_block"] = None

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"MatMulConfig M_block: {self.M_block}, K_block: {self.K_block}, N_block: {self.N_block}"
