"""Custom GUI widgets for FreeSimpleGUI."""

import FreeSimpleGUI as sg


def create_labeled_input(label: str, key: str, default: str = "", tooltip: str = "") -> list:
    """Helper to create labeled text input row."""
    return [
        sg.Text(label, size=(22, 1), tooltip=tooltip),
        sg.Input(default, key=key, tooltip=tooltip),
    ]
