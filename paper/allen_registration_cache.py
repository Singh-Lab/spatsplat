"""Reuse exact per-method fits when the report's comparison roster changes."""
from pathlib import Path

from .allen_registration_benchmark import cached_method as _cached_method, protocol_key
from .dlpfc_registration import pair_key


def cached_method(family, pair, config, root, figures, *, run=False):
    root = Path(root)
    key = pair_key(pair, dict(family=family, settings=config, implementation=protocol_key(config)))
    target = root/family/key/'alignment.npz'
    if not target.exists():
        # The per-method key includes inputs, settings, versions and fitting code.
        candidates = sorted(root.parent.parent.glob(f'*/{root.name}/{family}/{key}/alignment.npz'))
        if candidates:
            source = candidates[0]
            print('Reusing exact cached fit:', family, source, flush=True)
            return _cached_method(family, pair, config, source.parents[2], figures, run=False)
    return _cached_method(family, pair, config, root, figures, run=run)
