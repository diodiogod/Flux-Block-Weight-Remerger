"""
Usage:

Regular SVD: 
python svd_low_rank_lora.py --filename="How2Draw-V2_000002800.safetensors" \
    --new_rank=4 --new_lora_path="How2Draw-V2_000002800_svd.safetensors"

Randomized SVD:
python svd_low_rank_lora.py --filename="How2Draw-V2_000002800.safetensors" \
    --new_rank=4 --niter=5 --new_lora_path="How2Draw-V2_000002800_svd.safetensors"
"""

import torch
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
    # Store the original dtype and cast to float32
    original_dtype = matrix.dtype
    matrix = matrix.to(torch.float32)

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

    # Cast results back to the original dtype
    return U[:, :rank].to(original_dtype), S[:rank].to(original_dtype), Vh[:rank, :].to(original_dtype)



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
    # Store the original dtype and cast to float32
    dtype = lora_A.dtype
    lora_A = lora_A.to("cuda", torch.float32)
    lora_B = lora_B.to("cuda", torch.float32)

    # Compute the low-rank update matrix
    delta_W = lora_B @ lora_A

    # Perform SVD on the update matrix
    if niter is None:
        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
    else:
        U, S, Vh = randomized_svd(delta_W, rank=new_rank, niter=niter)

    # Keep only the top 'new_rank' singular values and vectors
    U_new = U[:, :new_rank]
    S_new = S[:new_rank]
    Vh_new = Vh[:new_rank, :]

    # Compute the square roots of the singular values
    S_sqrt = torch.sqrt(S_new)

    # Compute the new lora_B and lora_A matrices
    lora_B_new = U_new * S_sqrt.unsqueeze(0)
    lora_A_new = S_sqrt.unsqueeze(1) * Vh_new

    # Cast back to the original dtype
    return lora_A_new.to(dtype), lora_B_new.to(dtype)



def reduce_lora_rank_state_dict(state_dict, niter, new_rank=4):
    """
    Reduces the rank of all LoRA matrices in the given state dict.

    Supports both `ai-toolkit` and `sd-scripts` formats.

    Args:
        state_dict (dict): The state dict containing LoRA matrices.
        niter (int): Number of power iterations for randomized SVD.
        new_rank (int): Desired lower rank.

    Returns:
        new_state_dict (dict): State dict with reduced-rank LoRA matrices.
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

                # Scale by alpha (sd-scripts uses scaled LoRA weights)
                scale = alpha.item() / lora_down.shape[0]
                lora_down *= scale
                lora_up *= scale

                # Debug: Print key being processed
                print(f"Processing key (sd-scripts): {lora_down_key}")

                # Apply rank reduction
                lora_down_new, lora_up_new = reduce_lora_rank(lora_down, lora_up, niter=niter, new_rank=new_rank)

                # Update state dict
                new_state_dict[lora_down_key] = lora_down_new
                new_state_dict[lora_up_key] = lora_up_new
                new_state_dict[alpha_key] = torch.scalar_tensor(new_rank, dtype=lora_down.dtype)

        elif not is_sd_scripts and ".lora_A.weight" in key:  # Handle ai-toolkit format
            # Find corresponding .lora_B.weight
            lora_A_key = key
            lora_B_key = key.replace(".lora_A.weight", ".lora_B.weight")

            if lora_B_key in state_dict:
                lora_A = state_dict[lora_A_key]
                lora_B = state_dict[lora_B_key]

                # Debug: Print key being processed
                print(f"Processing key (ai-toolkit): {lora_A_key}")

                # Apply rank reduction
                lora_A_new, lora_B_new = reduce_lora_rank(lora_A, lora_B, niter=niter, new_rank=new_rank)

                # Update state dict
                new_state_dict[lora_A_key] = lora_A_new
                new_state_dict[lora_B_key] = lora_B_new

    return new_state_dict




def compare_approximation_error(orig_state_dict, new_state_dict):
    """
    Compares the approximation error between the original and new state dicts.
    """
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
    filename: str,
    new_rank: int,
    niter: int = None,
    check_error: bool = False,
    new_lora_path: str = None,
):
    """
    Main function for reducing LoRA rank.
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

    # Reduce the rank of LoRA matrices
    print("Reducing LoRA rank...")
    try:
        new_state_dict = reduce_lora_rank_state_dict(original_state_dict, niter=niter, new_rank=new_rank)
    except Exception as e:
        print(f"Error during rank reduction: {e}")
        return

    # Optional: Compare the approximation error
    if check_error:
        print("Comparing approximation error...")
        try:
            compare_approximation_error(original_state_dict, new_state_dict)
        except Exception as e:
            print(f"Error during error comparison: {e}")
            return

    # Save the reduced LoRA weights
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
