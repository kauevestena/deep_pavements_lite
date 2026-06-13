"""
Backward compatibility shim.

New code should import from deep_pavements.constants instead:

    from deep_pavements.constants import DEVICE, default_surfaces
"""

from deep_pavements.constants import *  # noqa: F401, F403
