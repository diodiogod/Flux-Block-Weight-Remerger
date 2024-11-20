"""
Usage:

python upsample_lora_rank.py --repo_id="cocktailpeanut/optimus" \
    --filename="optimus.safetensors" \
    --new_lora_path="optimus_16.safetensors" \
    --new_rank=16
"""

import torch
from huggingface_hub import hf_hub_download
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

    # Perform QR decomposition
    Q, R = torch.linalg.qr(matrix.T, mode="reduced")  # Transpose to get [columns, original_rows]
    Q = Q.T  # Back to [original_rows, columns]

    # Generate orthogonal vectors
    if target_rows > original_rows:
        additional_rows = target_rows - original_rows
        random_matrix = torch.randn(additional_rows, cols, dtype=matrix.dtype, device=matrix.device)
        # Orthogonalize against existing Q
        for i in range(additional_rows):
            v = random_matrix[i]
            v = v - Q.T @ (Q @ v)
            v = v / v.norm()
            Q = torch.cat([Q, v.unsqueeze(0)], dim=0)
    extended_matrix = Q
    return extended_matrix


def increase_lora_rank_orthogonal(state_dict, target_rank=16):
    """
    Increases the rank of all LoRA matrices in the given state dict using orthogonal extension.

    Args:
        state_dict (dict): The state dict containing LoRA matrices.
        target_rank (int): Desired higher rank.

    Returns:
        new_state_dict (dict): State dict with increased-rank LoRA matrices.
    """
    new_state_dict = state_dict.copy()
    for key in state_dict.keys():
        if "lora_A.weight" in key:
            lora_A_key = key
            lora_B_key = key.replace("lora_A.weight", "lora_B.weight")
            if lora_B_key in state_dict:
                lora_A = state_dict[lora_A_key]
                dtype = lora_A.dtype
                lora_A = lora_A.to("cuda", torch.float32)
                lora_B = state_dict[lora_B_key]
                lora_B = lora_B.to("cuda", torch.float32)

                original_rank = lora_A.shape[0]

                # Extend lora_A and lora_B
                lora_A_new = orthogonal_extension(lora_A, target_rank).to(dtype)
                lora_B_new = orthogonal_extension(lora_B.T, target_rank).T.to(dtype)  # Transpose to match dimensions

                # Update the state dict
                new_state_dict[lora_A_key] = lora_A_new
                new_state_dict[lora_B_key] = lora_B_new

                print(
                    f"Increased rank of {lora_A_key} and {lora_B_key} from {original_rank} to {target_rank} using orthogonal extension"
                )

    return new_state_dict


def compare_approximation_error(orig_state_dict, new_state_dict):
    for key in orig_state_dict:
        if "lora_A.weight" in key:
            lora_A_key = key
            lora_B_key = key.replace("lora_A.weight", "lora_B.weight")
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
    repo_id: str,
    filename: str,
    new_rank: int,
    check_error: bool = False,
    new_lora_path: str = None,
):
    # ckpt_path = hf_hub_download(repo_id="TheLastBen/The_Hound", filename="sandor_clegane_single_layer.safetensors")
    if new_lora_path is None:
        raise ValueError("Please provide a path to serialize the converted state dict.")

    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    original_state_dict = safetensors.torch.load_file(ckpt_path)
    new_state_dict = increase_lora_rank_orthogonal(original_state_dict, target_rank=new_rank)

    if check_error:
        compare_approximation_error(original_state_dict, new_state_dict)

    new_state_dict = {k: v.to("cpu").contiguous() for k, v in new_state_dict.items()}
    # safetensors.torch.save_file(new_state_dict, "sandor_clegane_single_layer_32.safetensors")
    safetensors.torch.save_file(new_state_dict, new_lora_path)


if __name__ == "__main__":
    fire.Fire(main)
