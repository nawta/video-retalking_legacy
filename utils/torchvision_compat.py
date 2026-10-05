"""Compatibility shim for basicsr 1.4.2 on torchvision >= 0.17.

basicsr 1.4.2 imports `torchvision.transforms.functional_tensor`, which
torchvision removed in 0.17. The functions it needs (rgb_to_grayscale) are in
`torchvision.transforms.functional`. Import this module before anything that
imports basicsr, gfpgan, or facexlib.
"""
import sys

try:
    import torchvision.transforms.functional_tensor  # noqa: F401  (torchvision < 0.17)
except ImportError:
    import torchvision.transforms.functional as _functional
    sys.modules['torchvision.transforms.functional_tensor'] = _functional
