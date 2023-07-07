import torch

if torch.cuda.is_available():
    print("GPU is available.")
    print("GPU device count:", torch.cuda.device_count())
    print("Current GPU device:", torch.cuda.current_device())
    print("GPU device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("GPU is not available.")
    try:
        _ = torch.cuda.get_device_properties(0)
    except AssertionError as error:
        print(f"Error: {error}")
    except RuntimeError as error:
        print(f"Error: {error}")