import torch
import glob
import os
import numpy as np

def render_arc_grid(tokens):
    """Converts a flat sequence into a visual 30x30 ARC grid."""
    # ARC tokens: 0=Pad, 1=EOS/Boundary, 2-11=Colors 0-9
    grid = tokens.reshape(30, 30)
    lines = []
    for r in range(30):
        row_str = ""
        for c in range(30):
            t = grid[r, c].item()
            if t == 0: row_str += "  " # Empty space for padding
            elif t == 1: row_str += "X " # Boundary marker
            else: row_str += str(t - 2) + " " # The actual color
        if row_str.strip(): # Only add rows that aren't just padding
            lines.append(row_str)
    return "\n".join(lines)

def main():
    pred_files = glob.glob("checkpoints/**/*_all_preds.0", recursive=True)
    if not pred_files:
        print("No predictions found. Did the evaluation finish?")
        return

    latest_file = max(pred_files, key=os.path.getctime)
    print(f"Analyzing: {latest_file}\n")
    data = torch.load(latest_file, map_location="cpu")

    logits = data["logits"]
    labels = data["labels"]
    inputs = data["inputs"]
    preds = logits.argmax(dim=-1)

    num_puzzles = len(preds)
    exact_matches = 0
    total_tokens = 0
    correct_tokens = 0

    for i in range(num_puzzles):
        # 1. Calculate Pixel-Level (Token) stats
        mask = (labels[i] > 0) # Ignore padding
        if mask.sum() > 0:
            token_match = (preds[i][mask] == labels[i][mask])
            correct_tokens += token_match.sum().item()
            total_tokens += mask.sum().item()

            # 2. Calculate Grid-Level Exact Match (The "Official" Score)
            # If every non-padding token matches exactly, it's a win.
            if token_match.all():
                exact_matches += 1

    print("=" * 40)
    print(f"OFFICIAL ARC SCORE (Exact Match): {(exact_matches/num_puzzles)*100:.2f}%")
    print(f"({exact_matches} out of {num_puzzles} puzzles solved perfectly)")
    print("-" * 40)
    print(f"Token-Level Accuracy: {(correct_tokens/total_tokens)*100:.2f}%")
    print("=" * 40)

    # Visual Inspection of the first success (or just the first example)
    for i in range(min(1, num_puzzles)):
        print(f"\nVISUAL CHECK - PUZZLE {i}:")
        print("\nINPUT:")
        print(render_arc_grid(inputs[i]))
        print("\nMODEL PREDICTION:")
        print(render_arc_grid(preds[i]))
        print("\nGROUND TRUTH:")
        print(render_arc_grid(labels[i]))

if __name__ == "__main__":
    main()