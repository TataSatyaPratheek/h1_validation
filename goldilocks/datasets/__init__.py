"""
Goldilocks Datasets Package
===========================
Unified interface for all Goldilocks benchmark datasets.
"""
from .boolean import get_boolean_dataloader, BOOLEAN_TASKS, generate_parity, generate_majority, generate_hidden_parity
from .geometric import get_geometric_dataloader, GEOMETRIC_TASKS, generate_circle, generate_spiral
from .molecular import get_molecular_dataloader, MOLECULAR_TASKS

# Master registry of all tasks
ALL_TASKS = {
    **{f"boolean/{k}": v for k, v in BOOLEAN_TASKS.items()},
    **{f"geometric/{k}": v for k, v in GEOMETRIC_TASKS.items()},
    **{f"molecular/{k}": v for k, v in MOLECULAR_TASKS.items()},
}

__all__ = [
    "get_boolean_dataloader", "BOOLEAN_TASKS",
    "get_geometric_dataloader", "GEOMETRIC_TASKS",
    "get_molecular_dataloader", "MOLECULAR_TASKS",
    "ALL_TASKS",
]
