"""
Usage:

python low_rank_lora.py --repo_id=glif/how2draw --filename="How2Draw-V2_000002800.safetensors" \
    --new_rank=4 --new_lora_path="How2Draw-V2_000002800_rank_4.safetensors"
"""

import torch
from huggingface_hub import hf_hub_download
import safetensors.torch
import fire


def sparse_random_projection_matrix(original_rank, new_rank, density=0.1):
    """
    Generates a sparse random projection matrix.

    Args:
        original_rank (int): Original rank (number of rows).
        new_rank (int): Reduced rank (number of columns).
        density (float): Fraction of non-zero elements.

    Returns:
        R (torch.Tensor): Sparse random projection matrix.
    """
    R = torch.zeros(new_rank, original_rank)
    num_nonzero = int(density * original_rank)
    for i in range(new_rank):
        indices = torch.randperm(original_rank)[:num_nonzero]
        values = torch.randn(num_nonzero)
        R[i, indices] = values
    return R / torch.sqrt(torch.tensor(new_rank, dtype=torch.float32))


def reduce_lora_rank_random_projection(lora_A, lora_B, new_rank=4, use_sparse=False):
    """
    Reduces the rank of LoRA matrices lora_A and lora_B using random projections.

    Args:
        lora_A (torch.Tensor): Original lora_A matrix of shape [original_rank, in_features].
        lora_B (torch.Tensor): Original lora_B matrix of shape [out_features, original_rank].
        new_rank (int): Desired lower rank.
        use_sparse (bool): Use sparse projection matrix.

    Returns:
        lora_A_new (torch.Tensor): Reduced lora_A matrix of shape [new_rank, in_features].
        lora_B_new (torch.Tensor): Reduced lora_B matrix of shape [out_features, new_rank].
    """
    original_rank = lora_A.shape[0]  # Assuming lora_A.shape = [original_rank, in_features]

    # Generate random projection matrix
    if use_sparse:
        R = sparse_random_projection_matrix(original_rank=original_rank, new_rank=new_rank)
    else:
        R = torch.randn(new_rank, original_rank, dtype=torch.float32) / torch.sqrt(
            torch.tensor(new_rank, dtype=torch.float32)
        )
    R = R.to(lora_A.device, lora_A.dtype)

    # Project lora_A and lora_B
    lora_A_new = (R @ lora_A.to(R.dtype)).to(lora_A.dtype)  # Shape: [new_rank, in_features]
    lora_B_new = (lora_B.to(R.dtype) @ R.T).to(lora_B.dtype)  # Shape: [out_features, new_rank]

    return lora_A_new, lora_B_new


def reduce_lora_rank_state_dict_random_projection(state_dict, new_rank=4, use_sparse=False):
    """
    Reduces the rank of all LoRA matrices in the given state dict using random projections.

    Args:
        state_dict (dict): The state dict containing LoRA matrices.
        new_rank (int): Desired lower rank.
        use_sparse (bool): Use sparse projection matrix.

    Returns:
        new_state_dict (dict): State dict with reduced-rank LoRA matrices.
    """
    new_state_dict = state_dict.copy()
    keys = list(state_dict.keys())
    for key in keys:
        if "lora_A.weight" in key:
            # Find the corresponding lora_B
            lora_A_key = key
            lora_B_key = key.replace("lora_A.weight", "lora_B.weight")
            if lora_B_key in state_dict:
                lora_A = state_dict[lora_A_key]
                lora_B = state_dict[lora_B_key]

                # Ensure tensors are on CPU for random projection
                lora_A = lora_A.to("cuda")
                lora_B = lora_B.to("cuda")

                # Apply the rank reduction using random projection
                lora_A_new, lora_B_new = reduce_lora_rank_random_projection(
                    lora_A, lora_B, new_rank=new_rank, use_sparse=use_sparse
                )

                # Update the state dict
                new_state_dict[lora_A_key] = lora_A_new
                new_state_dict[lora_B_key] = lora_B_new

                print(f"Reduced rank of {lora_A_key} and {lora_B_key} to {new_rank}")

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
    use_sparse: bool = False,
    check_error: bool = False,
    new_lora_path: str = None,
):
    # ckpt_path = hf_hub_download(repo_id="glif/how2draw", filename="How2Draw-V2_000002800.safetensors")
    if new_lora_path is None:
        raise ValueError("Please provide a path to serialize the converted state dict.")
    
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    original_state_dict = safetensors.torch.load_file(ckpt_path)
    new_state_dict = reduce_lora_rank_state_dict_random_projection(
        original_state_dict, new_rank=new_rank, use_sparse=use_sparse
    )

    if check_error:
        compare_approximation_error(original_state_dict, new_state_dict)

    new_state_dict = {k: v.to("cpu") for k, v in new_state_dict.items()}
    # safetensors.torch.save_file(new_state_dict, "How2Draw-V2_000002800_reduced_sparse.safetensors")
    safetensors.torch.save(new_state_dict, new_lora_path)


if __name__ == "__main__":
    fire.Fire(main)
