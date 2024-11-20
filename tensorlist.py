# main.py

import safetensors.torch
import os
from rich.console import Console
from rich.table import Table
from rich.prompt import IntPrompt

console = Console()

def print_tensor_names(input_file):
    """Print tensor names from the specified safetensors file."""
    tensor_dict = safetensors.torch.load_file(input_file)
    for name in tensor_dict.keys():
        print(name)

def display_lora_files(folder):
    """Display LoRA files in the specified folder."""
    if not os.path.exists(folder):
        console.print(f"[bold red]The folder '{folder}' does not exist.[/bold red]")
        return None
    files = [f for f in os.listdir(folder) if f.endswith('.safetensors')]
    if not files:
        console.print(f"[bold yellow]No LoRA files found in the folder '{folder}'.[/bold yellow]")
        return None
    table = Table(title=f"Available LoRA Files in '{folder}'")
    for i, file in enumerate(files, 1):
        table.add_row(str(i), file)
    console.print(table)
    return files

if __name__ == "__main__":
    # Prompt user to select folder
    folder_choice = IntPrompt.ask(
        "Do you want to read from the INPUT (1) or OUTPUT (2) folder?", 
        choices=["1", "2"], 
        default="1"
    )
    folder_path = "INPUT" if folder_choice == 1 else "OUTPUT"

    # Debug log the chosen folder path
    console.print(f"[bold cyan]Selected folder: {folder_path}[/bold cyan]")

    # Display available LoRA files
    files = display_lora_files(folder_path)
    if files is None:
        exit()

    # Ask user to select a file
    selected_file = None
    while selected_file is None:
        try:
            file_index = IntPrompt.ask("\nEnter the number of the LoRA file you want to process") - 1
            if 0 <= file_index < len(files):
                selected_file = files[file_index]
            else:
                console.print("[bold red]Invalid selection. Please enter a valid number.[/bold red]")
        except ValueError:
            console.print("[bold red]Please enter a valid number.[/bold red]")

    # Display tensor names
    input_file = os.path.join(folder_path, selected_file)
    console.print(f"\n[bold green]Tensor names in '{selected_file}':[/bold green]")
    print_tensor_names(input_file)
