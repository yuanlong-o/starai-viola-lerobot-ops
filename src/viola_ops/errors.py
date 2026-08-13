"""Domain errors shown directly to Viola operators."""


class ViolaOpsError(RuntimeError):
    """Base class for an actionable, fail-closed operator error."""


class ValidationError(ViolaOpsError):
    """Raised when supplied evidence or immutable inputs are invalid."""


class SafetyGateError(ViolaOpsError):
    """Raised before hardware construction when motion is not authorized."""
