import safetensors.torch
import torch
import os
from rich.console import Console
from rich.prompt import Prompt, FloatPrompt, IntPrompt

# Importing external scripts
from upsample_lora_rank_modified import increase_lora_rank_orthogonal
from svd_low_rank_lora_modified import reduce_lora_rank_state_dict as reduce_rank_svd
from low_rank_lora_modified import reduce_lora_rank_state_dict_random_projection as reduce_rank_random

console = Console()

# Citation for Sayak Paul's work
CITATION = """
This script integrates rank resizing methods from Sayak Paul's project:
https://huggingface.co/sayakpaul/flux-lora-resizing
"""

def display_lora_files(input_folder):
    """Displays the available LoRA files in the INPUT folder in a numbered list."""
    files = [f for f in os.listdir(input_folder) if f.endswith('.safetensors')]
    if not files:
        console.print(f"[bold red]No .safetensors files found in {input_folder}.[/bold red]")
        return None

    # Display the list of files with indices
    console.print("\n[bold cyan]Available LoRA files in the INPUT folder:[/bold cyan]")
    for idx, file in enumerate(files, start=1):
        console.print(f"[green]{idx}.[/green] {file}")

    return files


def generate_unique_filename(output_folder, merge_type, base_name):
    """
    Automatically generates a unique filename with the merge type at the beginning.
    Example: 'mrg_slice_01_LoRaOriginalName.safetensors'
    """
    base_output_file = os.path.join(output_folder, f"{merge_type}_01_{base_name}.safetensors")
    counter = 1

    # Ensure filename is unique by incrementing the counter
    while os.path.exists(base_output_file):
        base_output_file = os.path.join(output_folder, f"{merge_type}_{counter:02d}_{base_name}.safetensors")
        counter += 1

    return base_output_file

def adjust_rank(state_dict, rank_option, target_rank, niter=5):
    """
    Adjusts the rank of a LoRA state dict.
    :param state_dict: The state dict of the LoRA.
    :param rank_option: "increase", "svd", or "random".
    :param target_rank: The target rank.
    :param niter: Number of iterations for randomized SVD (default: 5).
    :return: The adjusted state dict.
    """
    if rank_option == "increase":
        console.print(f"[bold cyan]Increasing rank to {target_rank}...[/bold cyan]")
        return increase_lora_rank_orthogonal(state_dict, target_rank=target_rank)
    elif rank_option == "svd":
        console.print(f"[bold cyan]Reducing rank to {target_rank} using SVD (accurate)...[/bold cyan]")
        return reduce_rank_svd(state_dict, niter=niter, new_rank=target_rank)
    elif rank_option == "random":
        console.print(f"[bold cyan]Reducing rank to {target_rank} using Random Projections (faster)...[/bold cyan]")
        return reduce_rank_random(state_dict, new_rank=target_rank)
    else:
        raise ValueError("Unknown rank adjustment option.")


def match_lora_ranks(state_dict_a, state_dict_b, target_rank, rank_option):
    """
    Ensures both LoRAs are adjusted to the same target rank.
    :param state_dict_a: State dict of the first LoRA.
    :param state_dict_b: State dict of the second LoRA.
    :param target_rank: Target rank for adjustment.
    :param rank_option: Rank adjustment method ("increase", "svd", "random").
    :return: Adjusted state dicts for both LoRAs.
    """
    console.print(f"[bold cyan]Matching ranks to {target_rank} using {rank_option} method...[/bold cyan]")
    state_dict_a = adjust_rank(state_dict_a, rank_option, target_rank)
    state_dict_b = adjust_rank(state_dict_b, rank_option, target_rank)
    return state_dict_a, state_dict_b

def merge_loras(
    lora_a_path, lora_b_path, weight_a, weight_b, output_path, resize_method, rank_option=None, target_rank=None
):
    """
    Merges two LoRA files by adjusting tensor dimensions if necessary and applying weights.

    :param lora_a_path: Path to the first LoRA file.
    :param lora_b_path: Path to the second LoRA file.
    :param weight_a: Weight to apply to the first LoRA's tensors.
    :param weight_b: Weight to apply to the second LoRA's tensors.
    :param output_path: Path to save the merged LoRA file.
    :param resize_method: Method to adjust tensor sizes ("slice" or "pad").
    :param rank_option: Rank adjustment option ("increase", "svd", or "random").
    :param target_rank: Target rank for adjustment.
    """
    console.print(CITATION)

    # Load tensors
    console.print(f"[bold cyan]Loading LoRA A from {lora_a_path}[/bold cyan]")
    lora_a = safetensors.torch.load_file(lora_a_path)
    console.print(f"[bold cyan]Loading LoRA B from {lora_b_path}[/bold cyan]")
    lora_b = safetensors.torch.load_file(lora_b_path)

    # Adjust ranks if specified
    if rank_option and target_rank:
        lora_a, lora_b = match_lora_ranks(lora_a, lora_b, target_rank, rank_option)

    # Merge LoRAs
    merged_lora = {}
    for key in lora_a.keys():
        if key in lora_b:
            merged_lora[key] = weight_a * lora_a[key] + weight_b * lora_b[key]
        else:
            merged_lora[key] = lora_a[key]
    for key in lora_b.keys():
        if key not in merged_lora:
            merged_lora[key] = lora_b[key]

    # Save the merged file
    safetensors.torch.save_file(merged_lora, output_path)
    console.print(f"[bold cyan]Merged LoRA saved to {output_path}[/bold cyan]")

def determine_dynamic_rank(state_dict_a, state_dict_b, strategy):
    """
    Determines the dynamic rank (lowest or highest) between two LoRA state_dicts.

    :param state_dict_a: State dict of the first LoRA.
    :param state_dict_b: State dict of the second LoRA.
    :param strategy: "lowest" or "highest".
    :return: The determined rank.
    """
    def get_rank(state_dict):
        ranks = [
            v.shape[0] for k, v in state_dict.items() if "lora_A.weight" in k or "lora_down.weight" in k
        ]
        return min(ranks) if strategy == "lowest" else max(ranks)

    rank_a = get_rank(state_dict_a)
    rank_b = get_rank(state_dict_b)

    return min(rank_a, rank_b) if strategy == "lowest" else max(rank_a, rank_b)

def main():
    input_folder = os.path.join(os.path.dirname(__file__), 'INPUT')
    output_folder = os.path.join(os.path.dirname(__file__), 'OUTPUT')

    # Ensure folders exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Display available files
    files = display_lora_files(input_folder)
    if files is None:
        exit()

    # User selects LoRAs
    if not files or len(files) < 2:
       console.print("[bold red]At least two LoRA files are required to perform a merge.[/bold red]")
       exit()

    # Show the files and allow the user to pick by number
    console.print("\n[bold cyan]Select the LoRA files to merge:[/bold cyan]")
    lora_a_idx = IntPrompt.ask("Enter the number for the first LoRA", default=1)
    lora_b_idx = IntPrompt.ask("Enter the number for the second LoRA", default=2)

    # Ensure valid indices
    if not (1 <= lora_a_idx <= len(files)) or not (1 <= lora_b_idx <= len(files)):
        console.print("[bold red]Invalid file selection. Please try again.[/bold red]")
        exit()

    lora_a_name = files[lora_a_idx - 1]
    lora_b_name = files[lora_b_idx - 1]

    lora_a_path = os.path.join(input_folder, lora_a_name)
    lora_b_path = os.path.join(input_folder, lora_b_name)

    # User inputs weights
    weight_a = FloatPrompt.ask(f"Enter the weight for LoRA A ({lora_a_name})", default=1.0)
    weight_b = FloatPrompt.ask(f"Enter the weight for LoRA B ({lora_b_name})", default=1.0)

    # Load LoRAs for rank determination
    lora_a = safetensors.torch.load_file(lora_a_path)
    lora_b = safetensors.torch.load_file(lora_b_path)

    # Rank adjustment
    adjust_rank = Prompt.ask("Do you want to adjust the rank? (yes or no)", choices=["yes", "no"], default="no") == "yes"
    rank_option, target_rank, niter = None, None, 5
    if adjust_rank:
        rank_type = Prompt.ask(
            "Choose rank adjustment: 'increase' for higher rank, 'reduce' for lower rank",
            choices=["increase", "reduce"],
        )

        if rank_type == "reduce":
            dynamic_rank = Prompt.ask(
                "Do you want to reduce all ranks to a specified value or adjust dynamically? ('specify', 'lowest', 'highest')",
                choices=["specify", "lowest", "highest"],
            )
            if dynamic_rank == "specify":
                target_rank = IntPrompt.ask("Enter the target rank", default=8)
                rank_option = Prompt.ask(
                    "Choose method for reduction: 'svd' for accuracy, 'random' for speed",
                    choices=["svd", "random"]
                )
                if rank_option == "svd":
                    niter = IntPrompt.ask("Enter the number of iterations for SVD", default=5)
            else:
                strategy = "lowest" if dynamic_rank == "lowest" else "highest"
                target_rank = determine_dynamic_rank(lora_a, lora_b, strategy)
                console.print(f"[bold cyan]Determined target rank: {target_rank} ({strategy} rank strategy)[/bold cyan]")
                rank_option = Prompt.ask(
                    "Choose method for reduction: 'svd' for accuracy, 'random' for speed",
                    choices=["svd", "random"]
                )
                if rank_option == "svd":
                    niter = IntPrompt.ask("Enter the number of iterations for SVD", default=5)
        else:
            rank_option = "increase"
            target_rank = IntPrompt.ask("Enter the target rank for increase", default=16)


    # Generate unique filename
    merge_type = f"mrg_{rank_option or 'default'}"
    output_file = generate_unique_filename(output_folder, merge_type, f"{lora_a_name}_and_{lora_b_name}")
    console.print(f"\n[bold yellow]Saving as '{output_file}'[/bold yellow]")

    # Merge LoRAs
    merge_loras(lora_a_path, lora_b_path, weight_a, weight_b, output_file, "slice", rank_option, target_rank)

    console.print("\n[bold cyan]Merge operation completed![/bold cyan]")

if __name__ == "__main__":
    main()
