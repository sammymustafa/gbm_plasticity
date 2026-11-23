"""gbmplast - glioblastoma plasticity research pipeline.

Three independently valid, interoperable analysis modules:

A. Longitudinal MRI progression prediction (LUMIERE).
B. Radiogenomics and pseudoprogression (UPENN-GBM, UCSD-PTGBM).
C. Longitudinal molecular tumour plasticity (GBM-CARE, optional GLASS).

Each module is self-contained: no module silently joins patients across
cohorts.  Cross-cohort operations are restricted to *schema* reuse via the
canonical tables in :mod:`gbmplast.schema`.
"""

__version__ = "0.1.0"

from gbmplast.config import Config, load_config  # noqa: F401
