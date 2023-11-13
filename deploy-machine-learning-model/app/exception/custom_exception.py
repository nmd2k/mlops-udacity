"""Customize exception handling"""

class CustomException(Exception):
    """Custome Exception"""
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
