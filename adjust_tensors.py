import safetensors.torch
import torch
import os
import csv
import datetime

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt

console = Console()

def filter_layers_by_keywords(tensor_name, selected_keywords):
    """
    Checks if the tensor name contains core keywords (e.g., lora_A, lora_B, lora_up, lora_down)
    and additional keywords (e.g., attn, norm, proj_out, proj_mlp).
    
    The logic works as follows:
    - If only 'lora_A' or 'lora_B' is selected, match layers containing those keywords.
    - If 'lora_A' and an additional keyword (like 'attn') are selected, match layers containing both.
    - If only additional keywords are selected, match layers containing any of those keywords.
    """

    # Map old keywords to new ones
    keyword_mapping = {
        "lora_A": ["lora_A", "lora_down"],
        "lora_B": ["lora_B", "lora_up"]
    }

    # Expand selected keywords based on the mapping
    expanded_keywords = []
    for kw in selected_keywords:
        if kw in keyword_mapping:
            expanded_keywords.extend(keyword_mapping[kw])
        else:
            expanded_keywords.append(kw)

    # Separate core keywords and additional keywords
    core_keywords = [kw for kw in expanded_keywords if 'lora_' in kw]
    additional_keywords = [kw for kw in expanded_keywords if 'lora_' not in kw]

    # Logic for filtering based on selected keywords
    if core_keywords:
        # If core keywords are present
        core_match = any(core_keyword in tensor_name for core_keyword in core_keywords)
        
        if additional_keywords:
            # If both core and additional keywords are selected, ensure both match
            additional_match = any(additional_keyword in tensor_name for additional_keyword in additional_keywords)
            return core_match and additional_match  # AND logic
        else:
            return core_match  # Only core keywords are checked

    # If no core keywords are selected, match any of the additional keywords (OR logic)
    return any(additional_keyword in tensor_name for additional_keyword in additional_keywords)


def filter_and_adjust_proj_blocks(input_file, output_file, block_values, target_keywords, remove_tensors=False):
    """Filters and adjusts tensors based on target keywords and optionally removes those set to zero."""
    
    if len(block_values) != 57:
        raise ValueError(f"Expected exactly 57 values but got {len(block_values)}. Please provide exactly 57 values.")
    
    tensor_dict = safetensors.torch.load_file(input_file)
    single_block_prefixes = ["transformer.single_transformer_blocks.", "lora_unet_single_blocks_"]
    double_block_prefixes = ["transformer.transformer_blocks.", "lora_unet_double_blocks_"]
    filtered_tensors = {}
    
    adjusted_tensors = 0  # Track number of adjusted tensors
    removed_tensors = 0   # Track number of removed tensors

    for name, tensor in tensor_dict.items():
        block_value = None

        try:
            # Check if the name belongs to a single transformer block
            if any(prefix in name for prefix in single_block_prefixes):
                if "transformer.single_transformer_blocks." in name:
                    split_name = name.split('.')
                    block_num = int(split_name[2])  # Extract block number
                else:
                    split_name = name.split('_')
                    block_num = int(split_name[4])  # Extract block number

                if 0 <= block_num <= 37:
                    block_value = block_values[19 + block_num]  # Single blocks start after double blocks

            # Check if the name belongs to a double transformer block
            elif any(prefix in name for prefix in double_block_prefixes):
                if "transformer.transformer_blocks." in name:
                    split_name = name.split('.')
                    block_num = int(split_name[2])  # Extract block number
                else:
                    split_name = name.split('_')
                    block_num = int(split_name[4])  # Extract block number

                if 0 <= block_num <= 18:
                    block_value = block_values[block_num]  # Double blocks

        except ValueError:
            console.print(f"[bold red]Non-numeric block index detected in: {name}. Skipping block value extraction.[/bold red]")
            block_value = None

        # Process blocks based on the selected target keywords (match any keyword, not all)
        if block_value is not None and filter_layers_by_keywords(name, target_keywords):
            if block_value == 0:
                if remove_tensors:
                    console.print(f"[bold red]Removed {name}[/bold red]")
                    removed_tensors += 1  # Track removed tensors
                    continue  # Skip adding this tensor to the filtered dict (i.e., remove it)
                else:
                    console.print(f"[bold red]Set {name} to zero strength (not removed just set to 0)[/bold red]")
                    tensor = torch.zeros_like(tensor)  # Set tensor to zeros
                    adjusted_tensors += 1  # Track adjusted tensors
            elif block_value < 1:
                console.print(f"[bold yellow]Adjusted {name} by a factor of {block_value}[/bold yellow]")
                tensor = tensor * block_value  # Adjust the weight
                adjusted_tensors += 1  # Track adjusted tensors

            filtered_tensors[name] = tensor
        else:
            # Keep non-targeted layers unchanged
            filtered_tensors[name] = tensor

    # Report the results
    console.print(f"[green]{len(filtered_tensors)} tensors kept out of {len(tensor_dict)}.")
    console.print(f"[green]Filtered and adjusted {adjusted_tensors + removed_tensors} tensors "
                  f"({adjusted_tensors} adjusted, {removed_tensors} removed).")
    
    safetensors.torch.save_file(filtered_tensors, output_file)
    console.print(f"[bold green]Filtered and adjusted tensors saved to {output_file}.[/bold green]")

    # Logging the process to CSV
    log_file = "log.csv"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lora_name = os.path.basename(output_file)  # Use the saved LoRA file name (output_file)
    target_filter = ", ".join(target_keywords)  # Convert target keywords to a readable string
    
    # Check if log.csv exists, if not, create it with the header
    log_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline='') as csvfile:
        log_writer = csv.writer(csvfile, delimiter=',')
        if not log_exists:
            # Write header
            log_writer.writerow(["Date", "LoRA File", "Block Values", "Target Filter", "Mode (Adjusted/Removed)", "Adjusted Tensors", "Removed Tensors"])
        # Write the log entry
        log_writer.writerow([current_time, lora_name, block_values, target_filter, "Removed" if remove_tensors else "Adjusted", adjusted_tensors, removed_tensors])

    console.print(f"[bold cyan]Log updated: {log_file}[/bold cyan]")
    
def parse_block_values(block_values_str):
    try:
        block_values = [float(value) for value in block_values_str.split(",")]
        return block_values
    except ValueError:
        raise ValueError("Invalid input. Ensure all block values are numbers (0, 1, or a float).")

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

def load_presets_from_file():
    """Loads presets from the preset_options.txt file using tab-separated values."""
    presets = {}
    preset_file = os.path.join(os.path.dirname(__file__), "preset_options.txt")
    
    if os.path.exists(preset_file):
        with open(preset_file, "r") as file:
            preset_num = 2  # Start numbering from 2, reserving 1 for custom input
            for line in file:
                parts = line.strip().split("\t")  # Split by tab character
                if len(parts) >= 3:  # Ensure the format is valid
                    preset_name = parts[1]  # Preset name
                    preset_values = parts[2]  # Block values as a string (already comma-separated in the file)
                    presets[str(preset_num)] = (preset_name, preset_values)
                    preset_num += 1  # Increment the preset number for the next preset
    return presets


def get_block_values():
    """Prompt the user for block values or preset selection, supporting both 19 and 57 block inputs."""

    # Load dynamic presets from file
    presets = load_presets_from_file()

    # Display option for custom input as "1"
    console.print("\n[bold cyan]Available Presets:[/bold cyan]")
    console.print("[bold green]1.[/bold green] Custom Values")

    # Print all dynamically loaded presets starting from "2"
    for preset_num, (preset_name, _) in presets.items():
        console.print(f"[bold green]{preset_num}.[/bold green] {preset_name}")

    # Default to "1" (Custom Values) if the user presses enter
    preset_choice = Prompt.ask("\nChoose a preset or enter [bold yellow]1[/bold yellow] for custom input", 
                               choices=["1"] + list(presets.keys()), default="1")

    if preset_choice in presets:
        # Parse the block values from the selected preset
        preset_name, preset_values = presets[preset_choice]
        
        # Debug print: Show the retrieved values before processing
        console.print(f"\n[bold cyan]Selected Preset:[/bold cyan] {preset_name}")
        console.print(f"[bold green]Using Block Values:[/bold green] {preset_values}")
        
        return parse_block_values(preset_values)
    else:
        # If "1" is selected or defaulted, prompt for custom input
        while True:
            block_values_str = Prompt.ask("\nEnter the 19 or 57 block values separated by commas (e.g., 1,1,0.5,...): ")
            try:
                block_values_input = [float(x.strip()) for x in block_values_str.split(",")]
                num_values = len(block_values_input)
                if num_values not in [19, 57]:
                    console.print(f"[bold red]Expected 19 or 57 values, but got {num_values}. Please try again.[/bold red]")
                    continue

                if num_values == 57:
                    return block_values_input
                elif num_values == 19:
                    block_values = [0] * 57
                    for i in range(19):
                        block_values[i] = block_values_input[i]
                    for i, val in enumerate(block_values_input):
                        block_values[19 + i * 2] = val
                        block_values[20 + i * 2] = val

                    console.print(f"\n[bold yellow]This will proceed with the corresponding 57 values:[/bold yellow] {','.join(map(lambda x: str(int(x)) if x.is_integer() else str(x), block_values))}")
                    return block_values
            except ValueError:
                console.print("[bold red]Invalid input. Please enter numeric values separated by commas.[/bold red]")


def select_target_keywords():
    """Prompt the user to select which layer types to target."""
    options = {
        "1": "lora_A",
        "2": "lora_B",
        "3": "attn",
        "4": "proj_mlp",
        "5": "proj_out",
        "6": "norm",
        "7": ".alpha"
    }

    console.print("\n[bold cyan]Select Target Layers:[/bold cyan]")
    console.print("[bold green]1.[/bold green] lora_A or lora_down")
    console.print("[bold green]2.[/bold green] lora_B or lora_up (default)")
    console.print("[bold green]3.[/bold green] attn")
    console.print("[bold green]4.[/bold green] proj_mlp")
    console.print("[bold green]5.[/bold green] proj_out")
    console.print("[bold green]6.[/bold green] norm")
    console.print("[bold green]7.[/bold green] .alpha")

    selections = Prompt.ask("\nSelect the layer types to target (comma-separated list, e.g., 2,3 for lora_B and attn)", default="2")
    selected_keywords = [options[opt.strip()] for opt in selections.split(",") if opt.strip() in options]

    if not selected_keywords:
        selected_keywords = ["lora_B"]  # Default to lora_B if no valid selection made

    console.print(f"\n[bold green]Selected Keywords: {', '.join(selected_keywords)}[/bold green]")
    return selected_keywords
	

def generate_unique_filename(output_folder, base_name):
    """
    Automatically generates a unique filename by appending a number if the file already exists.
    For example, if 'adjusted_file.safetensors' exists, it will create 'adjusted01_file.safetensors', 
    then 'adjusted02_file.safetensors', etc.
    """
    base_output_file = os.path.join(output_folder, f"adjusted_{base_name}")
    output_file = base_output_file
    counter = 1

    while os.path.exists(output_file):
        output_file = os.path.join(output_folder, f"adjusted{counter:02d}_{base_name}")
        counter += 1

    return output_file

if __name__ == "__main__":

    while True:

        # Define the input and output folder paths
        input_folder = os.path.join(os.path.dirname(__file__), 'INPUT')
        output_folder = os.path.join(os.path.dirname(__file__), 'OUTPUT')

        # Ensure the OUTPUT folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Display available LoRA files in the INPUT folder
        files = display_lora_files(input_folder)
        if files is None:
            exit()

        # Ask user to select a LoRA file
        selected_file = None
        while selected_file is None:
            try:
                file_index = IntPrompt.ask("\nEnter the number of the LoRA file you want to adjust") - 1
                if 0 <= file_index < len(files):
                    selected_file = files[file_index]
                else:
                    console.print("[bold red]Invalid selection. Please enter a valid number.[/bold red]")
            except ValueError:
                console.print("[bold red]Please enter a valid number.[/bold red]")

        input_file = os.path.join(input_folder, selected_file)

        # Automatically generate a unique output filename
        output_file = generate_unique_filename(output_folder, selected_file)
        console.print(f"\n[bold yellow]Saving as '{output_file}'[/bold yellow]")

        # Ask the user for the 57 block values (use preset or custom input)
        block_values = get_block_values()

        # Ask the user if they want to remove tensors that are set to 0
        remove_tensors = Prompt.ask("\nDo you want to [bold red]remove tensors[/bold red] that are set to 0 (instead of setting their weights to zero)?", choices=["yes", "no"], default="no") == "yes"

        # Ask the user which keywords to target
        target_keywords = select_target_keywords()

        # Run the adjustment with or without removing tensors and based on selected keywords
        filter_and_adjust_proj_blocks(input_file, output_file, block_values, target_keywords, remove_tensors=remove_tensors)

        # Ask if the user wants to process another file
        process_another = Prompt.ask("\nDo you want to process another LoRA file?", choices=["yes", "no"], default="yes")
        if process_another == "no":
            break  # Exit the loop if the user chooses not to process another file