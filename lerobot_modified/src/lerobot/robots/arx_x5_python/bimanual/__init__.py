import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))


def find_first_specific_so_file(root_dir):
    """Walk ``root_dir`` and return the first file named ``arx_x5_python*.so``.

    Returns None when nothing matches (directory missing, empty, or no
    matching artifact — typically means the native extension hasn't been
    built on this machine).
    """
    if not os.path.isdir(root_dir):
        return None
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith('arx_x5_python') and filename.endswith('.so'):
                return os.path.join(dirpath, filename)
    return None


so_file = find_first_specific_so_file(
    os.path.join(current_dir, 'api', 'arx_x5_python')
)

# Graceful fallback when the compiled artifact is missing. Mirrors the pattern
# used by piper_sdk_interface.py (print hint + leave symbols as None) so that
# importing lerobot.robots on a machine without arx built doesn't crash the
# whole package registry. Downstream code that actually instantiates BimanualArm
# / SingleArm will fail later with a clear AttributeError/None, which is much
# easier to diagnose than the previous ``TypeError: stat: path should be str,
# not NoneType`` thrown from os.path.exists(None).
BimanualArm = None
SingleArm = None

if so_file is None or not os.path.exists(so_file):
    print(
        "arx_x5_python native .so not found under "
        f"{os.path.join(current_dir, 'api', 'arx_x5_python')}. "
        "Skipping arx bimanual registration — build the arx extension if you "
        "need arxx5_bimanual support."
    )
else:
    sys.path.append(os.path.dirname(so_file))
    try:
        from lerobot.robots.arx_x5_python.bimanual.script.single_arm import SingleArm  # noqa: F401
        from lerobot.robots.arx_x5_python.bimanual.script.dual_arm import BimanualArm  # noqa: F401
    except ImportError as exc:
        logging.warning(
            "arx_x5_python native .so found at %s but Python modules failed to import: %s",
            so_file, exc,
        )
        BimanualArm = None
        SingleArm = None

__all__ = ['BimanualArm', 'SingleArm']
