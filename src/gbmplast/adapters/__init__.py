"""Dataset adapters. Importing this package registers every adapter."""

from gbmplast.adapters.base import (  # noqa: F401
    CohortDescription,
    DatasetAdapter,
    SourceFile,
    available_adapters,
    get_adapter,
    register,
)
from gbmplast.adapters import (  # noqa: F401
    gbm_care,
    glass,
    institutional,
    lumiere,
    ucsd_ptgbm,
    upenn_gbm,
)
