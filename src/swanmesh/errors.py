"""Custom exception classes for swanmesh."""

class SwanMeshError(Exception):
    """Base exception class for all swanmesh errors."""

class ConfigurationError(SwanMeshError):
    """Raised when configuration validation or parsing fails."""

class DataInputError(SwanMeshError):
    """Raised when input files (domain, bathymetry, slope) are missing or invalid."""

class MeshingError(SwanMeshError):
    """Raised when Gmsh or mesh generation process encounters an error."""

class ExportError(SwanMeshError):
    """Raised when exporting to SWAN Triangle format fails."""

class ConversionError(SwanMeshError):
    """Raised when converting Gmsh .msh to SWAN format fails."""
