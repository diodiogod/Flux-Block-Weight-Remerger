import safetensors.torch

def print_tensor_names(input_file):
    tensor_dict = safetensors.torch.load_file(input_file)
    for name in tensor_dict.keys():
        print(name)

if __name__ == "__main__":
    input_file = "J:/Aitools/Ostris_tools/2/ai-toolkit/BW/In/sweaty_shirt_flux_v6_000002280.safetensors"  # Replace with your LoRA safetensors path
    print_tensor_names(input_file)
