"""Conformance suite: rankwise against pinned NumPy / SciPy / astropy.

"Identical" means same dtype, same shape and equal values under IEEE equality with
NaN == NaN (±0 compare equal), as the handoff defines it.
"""

import numpy as np
import pytest

import rankwise as rw

ISAS = rw.available_isas()


@pytest.fixture(params=ISAS)
def isa(request):
    """Run the test on every ISA this CPU supports (forced via rankwise.use_isa)."""
    with rw.use_isa(request.param):
        yield request.param


def pytest_report_header(config):
    import numpy, scipy, astropy
    return [
        f"rankwise {rw.__version__}; ISAs: {', '.join(ISAS)}",
        f"numpy {numpy.__version__}, scipy {scipy.__version__}, astropy {astropy.__version__}",
    ]
