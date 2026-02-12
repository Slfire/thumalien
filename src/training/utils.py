"""Utilitaires partagés pour l'entraînement des modèles."""

import torch
from loguru import logger


def get_device() -> torch.device:
    """Détecte le meilleur device disponible (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Device : Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Device : CUDA GPU")
    else:
        device = torch.device("cpu")
        logger.info("Device : CPU")
    return device


def get_lora_config(task_type, r: int, alpha: int, dropout: float):
    """Crée une configuration LoRA pour PEFT.

    Args:
        task_type: peft.TaskType (SEQ_CLS pour classification).
        r: Rang de la matrice LoRA.
        alpha: Facteur de mise à échelle.
        dropout: Taux de dropout.
    """
    from peft import LoraConfig

    return LoraConfig(
        task_type=task_type,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_lin", "v_lin"],
    )
