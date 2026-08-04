# ADR 0003: Choice of FreeSimpleGUI for Desktop Interface

## Context
A lightweight, user-friendly desktop GUI is required to configure and run meshing pipelines without command-line experience.

## Decision
We select `FreeSimpleGUI` as the desktop UI framework for `swanmesh-gui`.

## Consequences
- Single-window tabbed architecture with zero boilerplate.
- Keeps GUI completely separated from core library package (`swanmesh` vs `swanmesh_gui`).
- Multi-threaded execution via Python `threading` keeps GUI interactive during long meshing runs.
