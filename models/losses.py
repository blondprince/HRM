from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_LABEL_ID = -100

def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )

def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))

def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, weight: Optional[torch.Tensor] = None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    # FIX: Use labels >= 0 to robustly ignore any negative padding (-100, -102, etc.)
    valid_mask = labels >= 0
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    loss = -torch.where(valid_mask, prediction_logprobs, 0)
    
    # NEW: Apply the class weights to the loss
    if weight is not None:
        label_weights = weight[transformed_labels.to(torch.long)]
        loss = loss * label_weights

    return loss

def softmax_cross_entropy(logits, labels, ignore_index: int = -100, weight: Optional[torch.Tensor] = None):
    # Slice the weight tensor to match the exact number of vocab logits
    w = weight[:logits.shape[-1]] if weight is not None else None
    
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]), 
        labels.to(torch.long).view(-1), 
        weight=w,
        ignore_index=ignore_index, 
        reduction="none"
    ).view(labels.shape)

class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
        # --- NEW: Class-Weighted Loss Tensor ---
        # Allocating 100 slots to safely cover all colors and special tokens.
        weights = torch.ones(100)
        
        # Background (0) gets a tiny weight. The model gets almost no credit for guessing black.
        weights[0] = 0.05   
        
        # Colors (1-9) get a massive weight. Missing a shape is now 100x more painful.
        weights[1:10] = 5.0 
        
        # Register as buffer so it moves to GPU automatically alongside the model weights
        self.register_buffer('class_weights', weights)
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Correctness
        with torch.no_grad():
            # FIX: Catch any negative padding dynamically (solves the -102 bug)
            mask = labels >= 0
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        # Pass the class_weights down to the loss function
        raw_lm_loss = self.loss_fn(outputs["logits"], labels, ignore_index=-100, weight=self.class_weights)
        
        # Zero out the loss for padding tokens manually just to be completely bulletproof
        raw_lm_loss = torch.where(labels >= 0, raw_lm_loss, 0)
        
        lm_loss = (raw_lm_loss / loss_divisor).sum()
        
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()