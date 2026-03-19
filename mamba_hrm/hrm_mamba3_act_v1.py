from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


#
# Ensure the local Mamba-3 repository is importable when running
# from inside the HRM project directory.
#
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

from mamba.mamba_ssm.modules.mamba3 import Mamba3  # type: ignore[import]


@dataclass
class MambaHRMInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class MambaHRMCarry:
    inner_carry: MambaHRMInnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class MambaHRMConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Latent size
    hidden_size: int

    # Mamba-3 configuration for L and H modules
    mamba_d_state_L: int = 128
    mamba_d_state_H: int = 256
    mamba_expand_L: float = 2.0
    mamba_expand_H: float = 2.0
    mamba_headdim: int = 64
    mamba_ngroups: int = 1
    mamba_mimo_rank: int = 4

    # Positional encodings (kept for compatibility; not used directly here)
    pos_encodings: str

    rms_norm_eps: float = 1e-5

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"


class Mamba3LBlock(nn.Module):
    """
    L-Module: low-level perception using Mamba-3 MIMO.
    """

    def __init__(self, config: MambaHRMConfig) -> None:
        super().__init__()
        self.norm_eps = config.rms_norm_eps
        self.mamba = Mamba3(
            d_model=config.hidden_size,
            d_state=config.mamba_d_state_L,
            expand=config.mamba_expand_L,
            headdim=config.mamba_headdim,
            ngroups=config.mamba_ngroups,
            is_mimo=True,
            mimo_rank=config.mamba_mimo_rank,
            is_outproj_norm=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Residual Mamba-3 MIMO over token representations
        m_out = self.mamba(hidden_states)
        return rms_norm(hidden_states + m_out, variance_epsilon=self.norm_eps)


class Mamba3HBlock(nn.Module):
    """
    H-Module: high-level abstract reasoning using Mamba-3 SSM.
    Uses the SISO variant to focus capacity on recurrent latent updates.
    """

    def __init__(self, config: MambaHRMConfig) -> None:
        super().__init__()
        self.norm_eps = config.rms_norm_eps
        self.mamba = Mamba3(
            d_model=config.hidden_size,
            d_state=config.mamba_d_state_H,
            expand=config.mamba_expand_H,
            headdim=config.mamba_headdim,
            ngroups=config.mamba_ngroups,
            is_mimo=False,
            is_outproj_norm=True,
        )

    def forward(self, hidden_states: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        # Additively inject lower-level state, then run Mamba-3 SSM.
        hidden_states = hidden_states + injection
        m_out = self.mamba(hidden_states)
        return rms_norm(hidden_states + m_out, variance_epsilon=self.norm_eps)


class MambaHRMReasoningModule(nn.Module):
    """
    Generic module holding a stack of Mamba-3 blocks.
    For L we use Mamba3LBlock, for H we use Mamba3HBlock.
    """

    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, injection: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs  # unused, kept for HRM API symmetry
        for layer in self.layers:
            if isinstance(layer, Mamba3HBlock):
                hidden_states = layer(hidden_states, injection)
            else:
                hidden_states = layer(hidden_states)
        return hidden_states


class MambaHRMInner(nn.Module):
    """
    Inner latent-reasoning core of the Mamba-HRM architecture.

    Replaces HRM's Transformer-based L/H modules with:
      - L: Mamba-3 MIMO (fast context compression)
      - H: Mamba-3 SISO with outproj norm (abstract latent reasoning)
    """

    def __init__(self, config: MambaHRMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # Reasoning layers
        self.H_level = MambaHRMReasoningModule(
            layers=[Mamba3HBlock(self.config) for _ in range(self.config.H_layers)]
        )
        self.L_level = MambaHRMReasoningModule(
            layers=[Mamba3LBlock(self.config) for _ in range(self.config.L_layers)]
        )

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Q head special init (same as HRM)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return MambaHRMInnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: MambaHRMInnerCarry):
        return MambaHRMInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self, carry: MambaHRMInnerCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[MambaHRMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Latent pondering loop (gradient-free warm-up)
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                # L-module runs multiple passes of Mamba-3 MIMO to compress context
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        # L-level only sees (H state + inputs)
                        z_L = self.L_level(z_L, injection=z_H + input_embeddings)

                # H-module updates slower, over L-level state
                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, injection=z_L)

        assert not z_H.requires_grad and not z_L.requires_grad

        # Final 1-step gradient update (trainable pondering step)
        z_L = self.L_level(z_L, injection=z_H + input_embeddings)
        z_H = self.H_level(z_H, injection=z_L)

        # LM Outputs
        new_carry = MambaHRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_Mamba3ACTV1(nn.Module):
    """
    Drop-in replacement for HierarchicalReasoningModel_ACTV1 that
    uses Mamba-3 for both L- and H-level latent reasoning.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = MambaHRMConfig(**config_dict)
        self.inner = MambaHRMInner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return MambaHRMCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self, carry: MambaHRMCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[MambaHRMCarry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q (bootstrapping)
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]

                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt_logits,
                        torch.maximum(next_q_halt_logits, next_q_continue_logits),
                    )
                )

        return MambaHRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs

