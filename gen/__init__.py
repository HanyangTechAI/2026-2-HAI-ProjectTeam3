"""Midterm generation interfaces; real models and explicit mocks stay separate."""


class GenerationError(RuntimeError):
    """A model could not complete the requested generation."""


class FaceNotFoundError(GenerationError):
    """No usable face was detected in an input or generated image."""


class ModelUnavailableError(FileNotFoundError):
    """Required local model assets or runtime dependencies are unavailable."""
