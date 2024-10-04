import safetensors.torch

def print_tensor_names(input_file):
    tensor_dict = safetensors.torch.load_file(input_file)
    for name in tensor_dict.keys():
        print(name)

if __name__ == "__main__":
    input_file = "J:/Aitools/Flux-Block-Weight-Remerger/INPUT/Male Ass Flux v4c-000001.safetensors"  # Replace with your LoRA safetensors path
    print_tensor_names(input_file)
