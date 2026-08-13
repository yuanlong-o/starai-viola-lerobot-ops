"""Errors raised by the Viola handoff contract."""


class HandoffError(RuntimeError):
    """Base class for handoff failures."""


class BundleValidationError(HandoffError):
    """A bundle or one of its referenced artifacts failed validation."""


class EnvironmentValidationError(HandoffError):
    """The current repository or runtime cannot produce/accept evidence."""


class EvidenceError(HandoffError):
    """Required online evidence could not be recorded."""
