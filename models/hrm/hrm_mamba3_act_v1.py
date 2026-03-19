"""
Compatibility wrapper.

Your Mamba-HRM implementation was placed under `HRM/mamba_hrm/hrm_mamba3_act_v1.py`,
but `HRM/utils/functions.py` imports models with the hardcoded `models.` prefix.

This file re-exports the actual implementation so existing Hydra configs
(`config/arch/*.yaml`) continue to work.
"""

from mamba_hrm.hrm_mamba3_act_v1 import (  # noqa: F401
    MambaHRMConfig,
    HierarchicalReasoningModel_Mamba3ACTV1,
)

__all__ = [
    "MambaHRMConfig",
    "HierarchicalReasoningModel_Mamba3ACTV1",
]

