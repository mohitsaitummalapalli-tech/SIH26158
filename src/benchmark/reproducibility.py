"""Phase 3E.6 Environmental & Numerical Reproducibility Standards (R0–R3).

Enforces strict verification across the four levels of scientific reproducibility:
- R0_METADATA: Metadata existence, hardware environment logging, commit hash tracking.
- R1_NUMERICAL: Floating-point metrics invariant within declared numerical tolerance.
- R2_DETERMINISTIC: Exact structural/inlier index identity under fixed random seeds.
- R3_BITWISE: Bit-for-bit SHA-256 binary artifact hash equivalence.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Optional, Union
import numpy as np

from src.benchmark.models import (
    ContractViolationError,
    ReproducibilityLevel,
)


def verify_reproducibility_level(
    result_a: Any,
    result_b: Any,
    level: Union[ReproducibilityLevel, str],
    tolerance: Optional[float] = None,
) -> bool:
    """Verifies that two benchmark outputs satisfy the declared reproducibility tier.
    
    Args:
        result_a: Baseline benchmark output or metric dictionary.
        result_b: Replicated benchmark output or metric dictionary.
        level: Target reproducibility level (R0 to R3).
        tolerance: Numerical tolerance for R1 evaluation (defaults to 1e-5).
        
    Returns:
        True if the reproducibility contract is satisfied.
        
    Raises:
        ContractViolationError: If the outputs fail to meet the reproducibility standard.
    """
    if isinstance(level, str):
        level = ReproducibilityLevel(level)

    # ------------------------------------------------------------------------
    # R0_METADATA: Verify existence of required provenance and environment keys
    # ------------------------------------------------------------------------
    if level == ReproducibilityLevel.R0_METADATA:
        required_keys = {"software_commit", "dataset_id", "created_at_utc"}
        dict_a = result_a if isinstance(result_a, dict) else getattr(result_a, "__dict__", {})
        dict_b = result_b if isinstance(result_b, dict) else getattr(result_b, "__dict__", {})

        missing_a = required_keys - set(dict_a.keys())
        missing_b = required_keys - set(dict_b.keys())
        if missing_a or missing_b:
            raise ContractViolationError(
                f"Level R0 metadata violation: Missing provenance fields. In A: {missing_a}, In B: {missing_b}"
            )
        return True

    # ------------------------------------------------------------------------
    # R1_NUMERICAL: Numerical difference within declared tolerance
    # ------------------------------------------------------------------------
    elif level == ReproducibilityLevel.R1_NUMERICAL:
        tol = 1e-5 if tolerance is None else float(tolerance)

        def _extract_floats(obj: Any) -> np.ndarray:
            if isinstance(obj, (int, float)):
                return np.array([float(obj)])
            elif isinstance(obj, np.ndarray):
                return obj.astype(np.float64).flatten()
            elif isinstance(obj, (list, tuple)):
                return np.array([float(x) for x in obj], dtype=np.float64)
            elif isinstance(obj, dict):
                vals = []
                for k in sorted(obj.keys()):
                    v = obj[k]
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                    elif isinstance(v, (list, tuple, np.ndarray)):
                        vals.extend(_extract_floats(v))
                return np.array(vals, dtype=np.float64)
            else:
                raise TypeError(f"Cannot extract numerical values from object of type {type(obj).__name__}")

        floats_a = _extract_floats(result_a)
        floats_b = _extract_floats(result_b)

        if floats_a.shape != floats_b.shape:
            raise ContractViolationError(
                f"Level R1 numerical violation: Shape mismatch ({floats_a.shape} vs {floats_b.shape})."
            )

        max_diff = float(np.max(np.abs(floats_a - floats_b)))
        if max_diff > tol:
            raise ContractViolationError(
                f"Level R1 numerical violation: Maximum absolute difference ({max_diff:.8e}) "
                f"exceeds declared tolerance ({tol:.8e})."
            )
        return True

    # ------------------------------------------------------------------------
    # R2_DETERMINISTIC: Exact structural/inlier index and string/enum identity
    # ------------------------------------------------------------------------
    elif level == ReproducibilityLevel.R2_DETERMINISTIC:
        if isinstance(result_a, set) and isinstance(result_b, set):
            if result_a != result_b:
                raise ContractViolationError(
                    f"Level R2 deterministic violation: Set elements differ: {result_a ^ result_b}"
                )
            return True
        elif isinstance(result_a, (list, tuple)) and isinstance(result_b, (list, tuple)):
            if list(result_a) != list(result_b):
                raise ContractViolationError(
                    f"Level R2 deterministic violation: Sequences differ."
                )
            return True
        elif isinstance(result_a, dict) and isinstance(result_b, dict):
            if result_a != result_b:
                raise ContractViolationError(
                    f"Level R2 deterministic violation: Dictionaries differ."
                )
            return True
        else:
            if result_a != result_b:
                raise ContractViolationError(
                    f"Level R2 deterministic violation: Values not equal under fixed seed."
                )
            return True

    # ------------------------------------------------------------------------
    # R3_BITWISE: Exact binary/hash identity
    # ------------------------------------------------------------------------
    elif level == ReproducibilityLevel.R3_BITWISE:
        def _get_bytes(obj: Any) -> bytes:
            if isinstance(obj, bytes):
                return obj
            elif isinstance(obj, str):
                return obj.encode("utf-8")
            elif isinstance(obj, np.ndarray):
                return obj.tobytes()
            else:
                import json
                return json.dumps(obj, sort_keys=True).encode("utf-8")

        bytes_a = _get_bytes(result_a)
        bytes_b = _get_bytes(result_b)

        hash_a = hashlib.sha256(bytes_a).hexdigest()
        hash_b = hashlib.sha256(bytes_b).hexdigest()

        if hash_a != hash_b:
            raise ContractViolationError(
                f"Level R3 bitwise violation: Binary SHA-256 mismatch ({hash_a} != {hash_b})."
            )
        return True

    raise ValueError(f"Unknown reproducibility level: {level}")
