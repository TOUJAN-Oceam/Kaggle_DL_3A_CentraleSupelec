import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Nom du GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Pas de GPU, entrainement sur CPU par defaut")
    return device