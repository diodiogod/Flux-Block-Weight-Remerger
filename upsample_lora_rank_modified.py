"""
Usage:

python upsample_lora_rank.py --filename="optimus.safetensors" \
    --new_lora_path="optimus_16.safetensors" \
    --new_rank=16
"""

import torch
import safetensors.torch
import fire


def orthogonal_extension(matrix, target_rows):
    """
    Extends the given matrix to have target_rows rows by adding orthogonal rows.

    Args:
        matrix (torch.Tensor): Original matrix of shape [original_rows, columns].
        target_rows (int): Desired number of rows.

    Returns:
        extended_matrix (torch.Tensor): Matrix of shape [target_rows, columns].
    """
    original_rows, cols = matrix.shape
    assert target_rows >= original_rows, "Target rows must be greater than or equal to original rows."

    # Cast to float32 for QR decomposition
    original_dtype = matrix.dtype
    matrix = matrix.to(torch.float32)

    # Perform QR decomposition
    Q, R = torch.linalg.qr(matrix.T, mode="reduced")  # Transpose to get [columns, original_rows]
    Q = Q.T  # Back to [original_rows, columns]

    # Generate orthogonal vectors
    if target_rows > original_rows:
        additional_rows = target_rows - original_rows
        random_matrix = torch.randn(additional_rows, cols, dtype=torch.float32, device=matrix.device)
        # Orthogonalize against existing Q
        for i in range(additional_rows):
            v = random_matrix[i]
            v = v - Q.T @ (Q @ v)
            v = v / v.norm()
            Q = torch.cat([Q, v.unsqueeze(0)], dim=0)

    # Cast back to the original dtype
    extended_matrix = Q.to(original_dtype)
    return extended_matrix



def increase_lora_rank_orthogonal(state_dict, target_rank=16):
    """
    Increases the rank of all LoRA matrices in the given state dict using orthogonal extension.

    Supports both `ai-toolkit` and `sd-scripts` formats.

    Args:
        state_dict (dict): The state dict containing LoRA matrices.
        target_rank (int): Desired higher rank.

    Returns:
        new_state_dict (dict): State dict with increased-rank LoRA matrices.
    """
    new_state_dict = state_dict.copy()
    keys = list(state_dict.keys())

    # Detect format (ai-toolkit or sd-scripts)
    is_sd_scripts = any(".lora_down.weight" in key for key in keys)

    for key in keys:
        if is_sd_scripts and "lora_down.weight" in key:  # Handle sd-scripts format
            # Find corresponding .lora_up.weight and .alpha
            lora_down_key = key
            lora_up_key = key.replace("lora_down.weight", "lora_up.weight")
            alpha_key = key.replace("lora_down.weight", "alpha")

            if lora_up_key in state_dict:
                lora_down = state_dict[lora_down_key]
                lora_up = state_dict[lora_up_key]
                alpha = state_dict.get(alpha_key, torch.tensor(1.0))

                if lora_down.shape[0] >= target_rank:
                    print(f"Skipping {lora_down_key} and {lora_up_key} as their current rank ({lora_down.shape[0]}) is already >= target rank ({target_rank})")
                    continue

                # Scale by alpha (sd-scripts uses scaled LoRA weights)
                scale = alpha.item() / lora_down.shape[0]
                lora_down *= scale
                lora_up *= scale

                # Extend lora_down and lora_up
                lora_down_new = orthogonal_extension(lora_down, target_rank)
                lora_up_new = orthogonal_extension(lora_up.T, target_rank).T


                # Update state dict
                new_state_dict[lora_down_key] = lora_down_new
                new_state_dict[lora_up_key] = lora_up_new
                new_state_dict[alpha_key] = torch.scalar_tensor(target_rank, dtype=lora_down.dtype)

                print(f"Increased rank of {lora_down_key} and {lora_up_key} to {target_rank} (sd-scripts format)")

        elif not is_sd_scripts and ".lora_A.weight" in key:  # Handle ai-toolkit format
            # Find corresponding .lora_B.weight
            lora_A_key = key
            lora_B_key = key.replace(".lora_A.weight", ".lora_B.weight")

            if lora_B_key in state_dict:
                lora_A = state_dict[lora_A_key]
                lora_B = state_dict[lora_B_key]

                # Extend lora_A and lora_B
                lora_A_new = orthogonal_extension(lora_A, target_rank)
                lora_B_new = orthogonal_extension(lora_B.T, target_rank).T

                # Update state dict
                new_state_dict[lora_A_key] = lora_A_new
                new_state_dict[lora_B_key] = lora_B_new

                print(f"Increased rank of {lora_A_key} and {lora_B_key} to {target_rank} (ai-toolkit format)")

    return new_state_dict


def compare_approximation_error(orig_state_dict, new_state_dict):
    """
    Compares the approximation error between the original and new state dicts.
    """
    for key in orig_state_dict:
        if "lora_A.weight" in key or "lora_down.weight" in key:
            if "lora_A.weight" in key:
                lora_A_key = key
                lora_B_key = key.replace("lora_A.weight", "lora_B.weight")
            else:
                lora_A_key = key
                lora_B_key = key.replace("lora_down.weight", "lora_up.weight")

            lora_A_old = orig_state_dict[lora_A_key]
            lora_B_old = orig_state_dict[lora_B_key]
            lora_A_new = new_state_dict[lora_A_key]
            lora_B_new = new_state_dict[lora_B_key]

            # Original delta_W
            delta_W_old = (lora_B_old @ lora_A_old).to("cuda")

            # Approximated delta_W
            delta_W_new = lora_B_new @ lora_A_new

            # Compute the approximation error
            error = torch.norm(delta_W_old - delta_W_new, p="fro") / torch.norm(delta_W_old, p="fro")
            print(f"Relative error for {lora_A_key}: {error.item():.6f}")


def main(
    filename: str,
    new_rank: int,
    check_error: bool = False,
    new_lora_path: str = None,
):
    """
    Main function for increasing LoRA rank.
    """
    if new_lora_path is None:
        raise ValueError("Please provide a path to serialize the converted state dict.")

    print(f"Loading safetensors file from: {filename}")
    try:
        original_state_dict = safetensors.torch.load_file(filename)
        print("File loaded successfully.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Detect format
    is_sd_scripts = any(".lora_down.weight" in key for key in original_state_dict.keys())
    format_type = "sd-scripts" if is_sd_scripts else "ai-toolkit"
    print(f"Detected format: {format_type}")

    # Increase the rank of LoRA matrices
    print("Increasing LoRA rank...")
    try:
        new_state_dict = increase_lora_rank_orthogonal(original_state_dict, target_rank=new_rank)
    except Exception as e:
        print(f"Error during rank increase: {e}")
        return

    # Optional: Compare the approximation error
    if check_error:
        print("Comparing approximation error...")
        try:
            compare_approximation_error(original_state_dict, new_state_dict)
        except Exception as e:
            print(f"Error during error comparison: {e}")
            return

    # Save the increased-rank LoRA weights
    print(f"Saving new LoRA state dict to: {new_lora_path}")
    try:
        new_state_dict = {k: v.to("cpu").contiguous() for k, v in new_state_dict.items()}
        safetensors.torch.save_file(new_state_dict, new_lora_path)
        print(f"File saved successfully at {new_lora_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return


if __name__ == "__main__":
    fire.Fire(main)
