import torch
import glob
import os

def token_to_grid(tokens):
    """Translates raw model tokens back into readable ARC colors."""
    out = []
    for t in tokens:
        t = t.item()
        if t == 0: 
            out.append(".") # Pad token
        elif t == 1: 
            out.append("|") # End of Sequence / Boundary
        else:
            out.append(str(t - 2)) # ARC colors are shifted by +2 (0-9 become 2-11)
    
    # Wrap to 30 characters (ARCMaxGridSize) for easier reading
    grid_str = "".join(out)
    return "\n".join(grid_str[i:i+30] for i in range(0, len(grid_str), 30))

def main():
    # Find the newest _all_preds.0 file in your checkpoints folder
    pred_files = glob.glob("checkpoints/**/*_all_preds.0", recursive=True)
    if not pred_files:
        print("Could not find any predictions! Check your checkpoints folder.")
        return

    latest_file = max(pred_files, key=os.path.getctime)
    print(f"Loading results from: {latest_file}\n")

    # Load the predictions to CPU
    data = torch.load(latest_file, map_location="cpu")

    logits = data["logits"]
    labels = data["labels"]
    inputs = data["inputs"]

    # Convert probabilities to actual color predictions (argmax)
    preds = logits.argmax(dim=-1)

    # Calculate actual accuracy on valid tokens (ignoring '0' padding)
    valid_mask = (labels > 0)
    if valid_mask.sum() > 0:
        correct = ((preds == labels) & valid_mask).sum().item()
        total = valid_mask.sum().item()
        acc = (correct / total) * 100
        print(f"=== TRUE EVALUATION ACCURACY ===")
        print(f"Accuracy: {acc:.2f}% ({correct}/{total} valid tokens matched)\n")
    else:
        print("Labels were completely hidden by test_set_mode. Displaying visual predictions only.\n")

    # Show the first 3 examples
    print("=== SAMPLE PREDICTIONS ===")
    for i in range(min(3, len(preds))):
        print(f"\n--- EXAMPLE {i+1} ---")
        print("INPUT GRID:")
        print(token_to_grid(inputs[i]))
        
        print("\nMODEL PREDICTION:")
        print(token_to_grid(preds[i]))
        
        if valid_mask.sum() > 0:
            print("\nACTUAL TARGET:")
            print(token_to_grid(labels[i]))
        print("=" * 40)

if __name__ == "__main__":
    main()