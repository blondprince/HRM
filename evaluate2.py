from typing import List
import yaml
import os
import itertools
import torch
import torch.distributed as dist
import pydantic
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

# Importing core logic from your pretrain script
from pretrain import PretrainConfig, init_train_state, create_dataloader

class EvalConfig(pydantic.BaseModel):
    checkpoint: str
    save_outputs: List[str] = ["inputs", "labels", "puzzle_identifiers", "logits"]
    num_puzzles: int = 50 

def run_few_shot_eval(config, train_state, loader, num_puzzles_to_eval):
    model = train_state.model
    model.eval()
    
    total_puzzles = 0
    exact_matches = 0
    
    pbar = tqdm(itertools.islice(loader, num_puzzles_to_eval), total=num_puzzles_to_eval, desc="Few-Shot Evaluating")
    
    with torch.no_grad():
        for set_name, batch, _ in pbar:
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            num_sequences = batch["inputs"].shape[0]
            
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            
            # 1. PRIMING PASS
            for i in range(num_sequences - 1):
                example_sub_batch = {k: v[i:i+1] for k, v in batch.items()}
                ponder_step = 0
                while True:
                    carry, _, _, _, all_finish = model(
                        carry=carry, 
                        batch=example_sub_batch, 
                        return_keys=[]
                    )
                    ponder_step += 1
                    # INCREASED FROM 10 TO 40
                    if all_finish or ponder_step >= 40: 
                        break
            
            # 2. TEST PASS
            test_sub_batch = {k: v[-1:] for k, v in batch.items()}
            ponder_step = 0
            outputs = None
            while True:
                carry, _, _, outputs, all_finish = model(
                    carry=carry, 
                    batch=test_sub_batch, 
                    # --- CRITICAL FIX: Explicitly ask for logits ---
                    return_keys=["logits"] 
                )
                ponder_step += 1
                # INCREASED FROM 15 TO 40
                if all_finish or ponder_step >= 40: 
                    break
            
            # 3. SCORING
            if outputs is None or "logits" not in outputs:
                print(f"\nERROR: Model did not return logits! Found: {list(outputs.keys()) if outputs else 'None'}")
                break

            final_logits = outputs["logits"] 
            preds = final_logits.argmax(dim=-1).squeeze(0) 
            labels = test_sub_batch["labels"].squeeze(0)   
            
            mask = (labels > 1) 
            if mask.sum() > 0:
                is_correct = (preds[mask] == labels[mask]).all().item()
                if is_correct:
                    exact_matches += 1
                total_puzzles += 1
            
            if total_puzzles > 0:
                pbar.set_postfix({"EM_Score": f"{(exact_matches/total_puzzles)*100:.2f}%"})

    print(f"\n{'='*30}\nFINAL OFFICIAL SCORE: {(exact_matches/total_puzzles)*100:.2f}%")
    print(f"Puzzles Solved: {exact_matches} / {total_puzzles}\n{'='*30}")

def launch():
    eval_cfg = EvalConfig(**OmegaConf.to_container(OmegaConf.from_cli()))
    
    # Define local distributed params for a single-GPU setup
    RANK = 0
    WORLD_SIZE = 1
    torch.cuda.set_device(RANK)

    config_path = os.path.join(os.path.dirname(eval_cfg.checkpoint), "all_config.yaml")
    with open(config_path, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

    # --- THE FIX: Pass rank and world_size explicitly ---
    _, train_metadata = create_dataloader(
        config, 
        split="train", 
        test_set_mode=False, 
        epochs_per_iter=1,
        global_batch_size=1, 
        rank=RANK, 
        world_size=WORLD_SIZE
    )
    
    eval_loader, _ = create_dataloader(
        config, 
        split="test", 
        test_set_mode=True, 
        epochs_per_iter=1,
        global_batch_size=1, 
        rank=RANK, 
        world_size=WORLD_SIZE
    )

    # Initialize Model
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)
    
    print(f"Loading checkpoint: {eval_cfg.checkpoint}")
    state_dict = torch.load(eval_cfg.checkpoint, map_location="cuda")
    cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    train_state.model.load_state_dict(cleaned_state_dict, strict=True)
    
    print("Starting Official Few-Shot Evaluation...")
    run_shot_eval_count = eval_cfg.num_puzzles
    run_few_shot_eval(config, train_state, eval_loader, run_shot_eval_count)

if __name__ == "__main__":
    launch()