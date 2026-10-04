"""Runtime provenance helpers shared by scoring and operational checks."""
import json
from pathlib import Path


def resolve_inference_provenance(sidecar):
    """Resolve the environment that produced scores from a score sidecar."""
    if not isinstance(sidecar, dict):
        raise TypeError("score provenance sidecar must be a mapping")
    provenance = (sidecar["original_inference_provenance"]
                  if "original_inference_provenance" in sidecar else sidecar.get("provenance"))
    if not isinstance(provenance, dict):
        raise ValueError("score sidecar has no valid inference provenance")
    return provenance


def runtime_mismatches(expected, actual):
    """Return explicit Python and package version differences."""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return ["recorded or active runtime provenance is missing or malformed"]
    mismatches = []
    if not isinstance(expected.get("python"), str) or not isinstance(actual.get("python"), str):
        mismatches.append("recorded or active Python version is missing or malformed")
    elif expected["python"] != actual["python"]:
        mismatches.append(
            f"Python version: expected {expected.get('python')!r}, "
            f"found {actual.get('python')!r}"
        )
    expected_packages = expected.get("packages")
    actual_packages = actual.get("packages")
    if not isinstance(expected_packages, dict) or not isinstance(actual_packages, dict):
        mismatches.append("recorded or active package versions are missing or malformed")
    else:
        for name in sorted(set(expected_packages) | set(actual_packages)):
            if (name not in expected_packages or name not in actual_packages
                    or expected_packages[name] != actual_packages[name]):
                mismatches.append(
                    f"package {name}: expected {expected_packages.get(name, '<not recorded>')!r}, "
                    f"found {actual_packages.get(name, '<not recorded>')!r}"
                )
    return mismatches


def validate_saved_score_runtime(score_path, active):
    """Reject mismatched saved scores before model loading or checkpoint writes."""
    score_path = Path(score_path)
    completed = Path(str(score_path) + '.provenance.json')
    checkpoint = Path(str(score_path) + '.run.json')
    if completed.exists():
        recorded = resolve_inference_provenance(json.loads(completed.read_text(encoding='utf8')))
    elif checkpoint.exists():
        recorded = json.loads(checkpoint.read_text(encoding='utf8')).get('config')
    elif score_path.exists():
        raise ValueError('score file exists without checkpoint or completion provenance; preserve it and stop')
    else:
        return
    mismatches = runtime_mismatches(recorded, active)
    if mismatches:
        raise ValueError('saved score runtime differs from selected environment; preserve the output '
                         'and use a fresh score path: ' + '; '.join(mismatches))
