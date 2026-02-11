"""Live trading adapter -- real execution against exchange."""

from src.adapters.base import BaseAdapter


class LiveAdapter(BaseAdapter):
    """Placeholder for live trading mode."""

    def __init__(self):
        raise NotImplementedError("Live adapter not yet implemented")
