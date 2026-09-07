"""Tuner config generation package. Only config_generator survives the 2026-09-07 prune
(see git tag pre-prune-2026-09-07 for the removed pipeline/hotspot/planner modules)."""

from .config_generator import TunerConfigGenerator

__all__ = ['TunerConfigGenerator']
