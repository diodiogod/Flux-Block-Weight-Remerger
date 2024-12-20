# Flux-Block-Weight-Remerger

A Python tool to filter, adjust, and optionally remove Flux block weights from a LoRA.

> **Note:** I'm not a dev, so this implementation might be wrong. It's all basically ChatGPT being guided.

 ![example](./example.jpg)

## Features

- **Adjust Weights:** Adjust weights for blocks and layers according to provided values (19 or 57 comma separated format). see: https://github.com/nihedon/sd-webui-lora-block-weight/issues/2#issuecomment-2390632995
- **Zero-out Weights:** Optionally remove layers that have their weights set to zero.
  > I don't know if removing leads to problems. It defaults to keep all layers.
- **Filter Layers:** Select and adjust specific layers based on keywords: 'lora_B/lora_up', 'lora_A/lora_down', 'proj_mlp', 'proj_out', 'attn', 'norm' - the default is all 'lora_B' layers.
  > I have no idea if this is correct, but it's what gave me in my testings the closest result to changing the block weights in Forge or ComfyUi.
> [!IMPORTANT]
>  CHOOSE .alpha if your LoRA was trained with kohya_ss. I think it's the most accurate.
- **Presets:** You can save your presets on the preset_options.txt file. 19 or 57 format.
- **Log:** Keep track of all loras adjusted on a log.csv file.

## Automatic Installation for Windows

1. Download the Repository
2. Double click the "start_windows.bat" file

## Manual Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/diodiogod/Flux-Block-Weight-Remerger
   cd Flux-Block-Weight-Remerger
   ```

2. **Set Up a Virtual Environment**:
   Make sure you have Python 3.x installed. Then, create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Install Dependencies**:
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare Your LoRA Files**:
   Place your safetensors LoRA files in the `INPUT` folder, you can leave multiple LoRas there. The processed file will be saved in the `OUTPUT` folder.

2. **Run the Script**:
   Execute the script, no arguments needed:
   ```bash
   py adjust_tensors.py
   ```
## Changelog
- 04/10/24 : handles loras with TE trained by kohya (ignore TE, won't adjust them). Also the single and double layer have different names. I think this works. lora_A = lora_down and lora_B = lora_up
- 23/10/24 : Allows for higher value adjustments like 1.25. Also prints unadjested layers so you know they were filtered but not adjusted. start.bat file will "git pull" automatically.
