import safetensors.torch
import torch
import os
import csv
import datetime
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, FloatPrompt, IntPrompt

console = Console()

def display_lora_files(input_folder):
    """Displays the available LoRA files in the INPUT folder in a table."""
    files = [f for f in os.listdir(input_folder) if f.endswith('.safetensors')]
    if not files:
        console.print(f"[bold red]No .safetensors files found in {input_folder}.[/bold red]")
        return None
    
    table = Table(title="Available LoRA Files")
    table.add_column("Number", style="cyan", justify="center")
    table.add_column("File Name", style="cyan")

    for idx, file in enumerate(files):
        table.add_row(str(idx + 1), file)
    
    console.print(table)
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


def adjust_tensor_size(tensor, target_shape, method="slice"):
    """
    Adjusts a tensor's size to match the target shape using slicing or padding.

    :param tensor: The tensor to adjust.
    :param target_shape: The desired shape.
    :param method: The adjustment method ("slice" or "pad").
    :return: The adjusted tensor.
    """
    if method == "slice":
        return tensor[:target_shape[0], :target_shape[1]]
    elif method == "pad":
        padding = (0, max(0, target_shape[1] - tensor.shape[1]), 
                   0, max(0, target_shape[0] - tensor.shape[0]))
        return torch.nn.functional.pad(tensor, padding, "constant", 0)
    else:
        raise ValueError(f"Unknown method: {method}")

def merge_loras(lora_a_path, lora_b_path, weight_a, weight_b, output_path, resize_method):
    """
    Merges two LoRA files by adjusting tensor dimensions if necessary and applying weights.

    :param lora_a_path: Path to the first LoRA file.
    :param lora_b_path: Path to the second LoRA file.
    :param weight_a: Weight to apply to the first LoRA's tensors.
    :param weight_b: Weight to apply to the second LoRA's tensors.
    :param output_path: Path to save the merged LoRA file.
    :param resize_method: Method to adjust tensor sizes ("slice" or "pad").
    """
    console.print(f"[bold cyan]Loading LoRA A from {lora_a_path}[/bold cyan]")
    lora_a_tensors = safetensors.torch.load_file(lora_a_path)
    
    console.print(f"[bold cyan]Loading LoRA B from {lora_b_path}[/bold cyan]")
    lora_b_tensors = safetensors.torch.load_file(lora_b_path)

    merged_tensors = {}

    for tensor_name in lora_a_tensors.keys() | lora_b_tensors.keys():
        tensor_a = lora_a_tensors.get(tensor_name, None)
        tensor_b = lora_b_tensors.get(tensor_name, None)

        if tensor_a is not None and tensor_b is not None:
            if tensor_a.shape != tensor_b.shape:
                console.print(f"[bold yellow]Adjusting dimensions for {tensor_name} using {resize_method}[/bold yellow]")
                if tensor_a.numel() > tensor_b.numel():
                    tensor_a = adjust_tensor_size(tensor_a, tensor_b.shape, method=resize_method)
                else:
                    tensor_b = adjust_tensor_size(tensor_b, tensor_a.shape, method=resize_method)
            
            merged_tensors[tensor_name] = tensor_a * weight_a + tensor_b * weight_b
            console.print(f"[bold green]Merged {tensor_name}: {weight_a} * A + {weight_b} * B[/bold green]")
        elif tensor_a is not None:
            merged_tensors[tensor_name] = tensor_a * weight_a
            console.print(f"[bold yellow]Only in A {tensor_name}: {weight_a} * A[/bold yellow]")
        elif tensor_b is not None:
            merged_tensors[tensor_name] = tensor_b * weight_b
            console.print(f"[bold yellow]Only in B {tensor_name}: {weight_b} * B[/bold yellow]")

    console.print(f"[bold cyan]Saving merged LoRA to {output_path}[/bold cyan]")
    safetensors.torch.save_file(merged_tensors, output_path)
    console.print("[bold green]Merge completed successfully![/bold green]")

def log_merge_process(output_file, weight_a, weight_b, lora_a_name, lora_b_name, resize_method):
    """Logs the merge process to a CSV file."""
    log_file = "merge_log.csv"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_exists = os.path.isfile(log_file)
    
    with open(log_file, "a", newline='') as csvfile:
        log_writer = csv.writer(csvfile, delimiter=',')
        if not log_exists:
            log_writer.writerow(["Date", "Merged LoRA File", "LoRA A File", "LoRA B File", "Weight A", "Weight B", "Resize Method"])
        log_writer.writerow([current_time, output_file, lora_a_name, lora_b_name, weight_a, weight_b, resize_method])

    console.print(f"[bold cyan]Log updated: {log_file}[/bold cyan]")

if __name__ == "__main__":
    input_folder = os.path.join(os.path.dirname(__file__), 'INPUT')
    output_folder = os.path.join(os.path.dirname(__file__), 'OUTPUT')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = display_lora_files(input_folder)
    if files is None:
        exit()

    lora_a_index = IntPrompt.ask("\nEnter the number of the first LoRA file you want to merge", default=1) - 1
    lora_a_name = files[lora_a_index]
    lora_a_path = os.path.join(input_folder, lora_a_name)

    lora_b_index = IntPrompt.ask("\nEnter the number of the second LoRA file you want to merge", default=2) - 1
    lora_b_name = files[lora_b_index]
    lora_b_path = os.path.join(input_folder, lora_b_name)

    weight_a = FloatPrompt.ask(f"Enter the weight for LoRA A ({lora_a_name})", default=1.0)
    weight_b = FloatPrompt.ask(f"Enter the weight for LoRA B ({lora_b_name})", default=1.0)

    resize_method = Prompt.ask("Choose the resize method (slice or pad)", choices=["slice", "pad"], default="slice")

    # Generate a unique output filename for the merged LoRA
    output_file = generate_unique_filename(
        output_folder,
        f"mrg_{resize_method}",
        f"{lora_a_name.split('.')[0]}_and_{lora_b_name.split('.')[0]}"
    )
    console.print(f"\n[bold yellow]Saving as '{output_file}'[/bold yellow]")


    merge_loras(lora_a_path, lora_b_path, weight_a, weight_b, output_file, resize_method)
    log_merge_process(output_file, weight_a, weight_b, lora_a_name, lora_b_name, resize_method)

    console.print("\n[bold cyan]Merge operation completed![/bold cyan]")
