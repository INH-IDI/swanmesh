"""Base interface for meshing engines in swanmesh."""

from abc import ABC, abstractmethod

import pandas as pd

from swanmesh.domain import DomainModel
from swanmesh.size_field import MeshSizeField


class MeshResult:
    """Stores generated 2D triangular mesh data (nodes, triangles, boundary edges)."""

    def __init__(self, nodes: pd.DataFrame, triangles: pd.DataFrame, boundary_edges: list[tuple[int, int]]):
        self.nodes = nodes  # Columns: N, X, Y
        self.triangles = triangles  # Columns: ID, ELEMENT1, ELEMENT2, ELEMENT3
        self.boundary_edges = boundary_edges

class BaseMesher(ABC):
    """Abstract base class for 2D mesh generators."""

    @abstractmethod
    def generate_mesh(self, domain: DomainModel, size_field: MeshSizeField) -> MeshResult:
        """Generate 2D triangular mesh from domain geometry and size field."""
