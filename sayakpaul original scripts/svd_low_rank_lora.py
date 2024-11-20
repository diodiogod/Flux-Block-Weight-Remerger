"""
Usage:

Regular SVD: 
python svd_low_rank_lora.py --repo_id=glif/how2draw --filename="How2Draw-V2_000002800.safetensors" \
    --new_rank=4 --new_lora_path="How2Draw-V2_000002800_svd.safetensors"

Randomized SVD:
python svd_low_rank_lora.py --repo_id=glif/how2draw --filename="How2Draw-V2_000002800.safetensors" \
    --new_rank=4 --niter=5 --new_lora_path="How2Draw-V2_000002800_svd.safetensors"
"""

import torch
from huggingface_hub import hf_hub_download
import safetensors.torch
import fire


def randomized_svd(matrix, rank, niter=5):
    """
    Performs a randomized SVD on the given matrix.
    Args:
        matrix (torch.Tensor): The input matrix.
        rank (int): The target rank.
        niter (int): Number of iterations for power method.
    Returns:
        U (torch.Tensor), S (torch.Tensor), Vh (torch.Tensor)
    """
    # Step 1: Generate a random Gaussian matrix
    omega = torch.randn(matrix.size(1), rank, device=matrix.device)

    # Step 2: Form Y = A * Omega
    Y = matrix @ omega

    # Step 3: Orthonormalize Y using QR decomposition
    Q, _ = torch.linalg.qr(Y, mode="reduced")

    # Power iteration (optional, improves approximation)
    for _ in range(niter):
        Z = matrix.T @ Q
        Q, _ = torch.linalg.qr(matrix @ Z, mode="reduced")

    # Step 4: Compute B = Q^T * A
    B = Q.T @ matrix

    # Step 5: Compute SVD of the small matrix B
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)

    # Step 6: Compute U = Q * Ub
    U = Q @ Ub

    return U[:, :rank], S[:rank], Vh[:rank, :]


def reduce_lora_rank(lora_A, lora_B, niter, new_rank=4):
    """
    Reduces the rank of LoRA matrices lora_A and lora_B with SVD, supporting truncated SVD, too.

    Args:
        lora_A (torch.Tensor): Original lora_A matrix of shape [original_rank, in_features].
        lora_B (torch.Tensor): Original lora_B matrix of shape [out_features, original_rank].
        niter (int): Number of power iterations for randomized SVD.
        new_rank (int): Desired lower rank.

    Returns:
        lora_A_new (torch.Tensor): Reduced lora_A matrix of shape [new_rank, in_features].
        lora_B_new (torch.Tensor): Reduced lora_B matrix of shape [out_features, new_rank].
    """
    # Compute the low-rank update matrix
    dtype = lora_A.dtype
    lora_A = lora_A.to("cuda", torch.float32)
    lora_B = lora_B.to("cuda", torch.float32)
    delta_W = lora_B @ lora_A

    # Perform SVD on the update matrix
    if niter is None:
        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
    # Perform randomized SVD
    if niter:
        U, S, Vh = randomized_svd(delta_W, rank=new_rank, niter=niter)

    # Keep only the top 'new_rank' singular values and vectors
    U_new = U[:, :new_rank]
    S_new = S[:new_rank]
    Vh_new = Vh[:new_rank, :]

    # Compute the square roots of the singular values
    S_sqrt = torch.sqrt(S_new)

    # Compute the new lora_B and lora_A matrices
    lora_B_new = U_new * S_sqrt.unsqueeze(0)  # Shape: [out_features, new_rank]
    lora_A_new = S_sqrt.unsqueeze(1) * Vh_new  # Shape: [new_rank, in_features]

    return lora_A_new.to(dtype), lora_B_new.to(dtype)


def reduce_lora_rank_state_dict(state_dict, niter, new_rank=4):
    """
    Reduces the rank of all LoRA matrices in the given state dict.

    Args:
        state_dict (dict): The state dict containing LoRA matrices.
        niter (int): Number of power iterations for ranodmized SVD.
        new_rank (int): Desired lower rank.

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

                # Apply the rank reduction
                lora_A_new, lora_B_new = reduce_lora_rank(lora_A, lora_B, niter=niter, new_rank=new_rank)

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
    niter: int = None,
    check_error: bool = False,
    new_lora_path: str = None,
):
    # ckpt_path = hf_hub_download(repo_id="glif/how2draw", filename="How2Draw-V2_000002800.safetensors")
    if new_lora_path is None:
        raise ValueError("Please provide a path to serialize the converted state dict.")

    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    original_state_dict = safetensors.torch.load_file(ckpt_path)
    new_state_dict = reduce_lora_rank_state_dict(original_state_dict, niter=niter, new_rank=new_rank)

    if check_error:
        compare_approximation_error(original_state_dict, new_state_dict)

    new_state_dict = {k: v.to("cpu").contiguous() for k, v in new_state_dict.items()}
    # safetensors.torch.save_file(new_state_dict, "How2Draw-V2_000002800_reduced_sparse.safetensors")
    safetensors.torch.save_file(new_state_dict, new_lora_path)


if __name__ == "__main__":
    fire.Fire(main)
