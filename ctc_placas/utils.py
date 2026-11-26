import torch
import torch.nn as nn


def save_model(model, path_dir, name_file):
    """Salva o modelo treinado em um arquivo."""
    torch.save(model.state_dict(), f"{path_dir}/{name_file}")


def load_model(model, path_dir, name_file):
    """Carrega o modelo treinado de um arquivo."""
    model.load_state_dict(torch.load(f"{path_dir}/{name_file}"))
    return model