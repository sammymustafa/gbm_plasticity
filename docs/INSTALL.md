# Installation

## Python version

**Python 3.10 or 3.11.** Not 3.12+, and the constraint comes from PyRadiomics:

- PyRadiomics 3.1.0 publishes wheels only up to `cp39`, so on any newer
  interpreter it must build from its source distribution.
- Its build runs the vendored `versioneer`, which calls
  `configparser.SafeConfigParser` - **removed in Python 3.12**. The build
  fails there with `AttributeError: module 'configparser' has no attribute
  'SafeConfigParser'`.

Everything else in the project runs on 3.12+; only the imaging extra is
pinned by this.

## The two PyRadiomics problems, and the fixes

### 1. The sdist omits two C headers

`pyradiomics-3.1.0.tar.gz` ships `radiomics/src/*.c` but **not** the matching
`cmatrices.h` and `cshape.h`, so the compile fails with:

```
radiomics/src/_cmatrices.c:6:10: fatal error: cmatrices.h: No such file or directory
```

Fetch them from the tagged release:

```bash
curl -sSL -o cmatrices.h https://raw.githubusercontent.com/AIM-Harvard/pyradiomics/v3.1.0/radiomics/src/cmatrices.h
```

```bash
curl -sSL -o cshape.h https://raw.githubusercontent.com/AIM-Harvard/pyradiomics/v3.1.0/radiomics/src/cshape.h
```

### 2. The sdist self-reports the wrong version

With no git metadata, versioneer falls back and reports `3.0.1a1`, so
`pip install pyradiomics==3.1.0` fails with *"has inconsistent version"*.
Installing from an **extracted directory named `pyradiomics-3.1.0`** lets
versioneer's `parentdir_prefix` resolve the version correctly.

### Putting it together

```bash
pip install "numpy==1.26.4" "setuptools<70" wheel versioneer==0.29
```

```bash
curl -sSL -o /tmp/pyradiomics.tar.gz https://files.pythonhosted.org/packages/03/c1/20fc2c50ab1e3304da36d866042a1905a2b05a1431ece35448ab6b4578f2/pyradiomics-3.1.0.tar.gz
```

```bash
mkdir -p /tmp/pyrad && tar xzf /tmp/pyradiomics.tar.gz -C /tmp/pyrad
```

```bash
curl -sSL -o /tmp/pyrad/pyradiomics-3.1.0/radiomics/src/cmatrices.h https://raw.githubusercontent.com/AIM-Harvard/pyradiomics/v3.1.0/radiomics/src/cmatrices.h
```

```bash
curl -sSL -o /tmp/pyrad/pyradiomics-3.1.0/radiomics/src/cshape.h https://raw.githubusercontent.com/AIM-Harvard/pyradiomics/v3.1.0/radiomics/src/cshape.h
```

```bash
pip install --no-build-isolation /tmp/pyrad/pyradiomics-3.1.0
```

Verify - `radiomics.__version__` reports `v3.1.0` even though pip records
`3.0.1a1`:

```bash
python -c "import radiomics; print(radiomics.__version__)"
```

## NumPy

Pinned to `<2.0`. PyRadiomics' C extensions are built against the NumPy 1.x
C API, and `scikit-survival` / `scanpy` in the pinned versions expect the same
ABI.

## Everything else

```bash
pip install -e ".[all]"
```

```bash
pip install -r requirements.txt
```

`requirements.txt` is the exact pinned set used to produce the results in
`results/`.

## Optional components

| Component | Needed for | If absent |
|---|---|---|
| `pyradiomics` | de-novo feature extraction | published feature tables still work; `gbmplast features` reports nothing |
| `rdata` | reading GBM-CARE `.RDS` matrices | Module C pseudobulk cannot run |
| `scikit-survival` | penalised Cox | `gbmplast model survival` reports it is unavailable |
| `shap` | SHAP attributions | permutation importance is used instead |
| `jupytext`, `jupyter` | notebooks | the `.py` sources are still readable |

## Network access

Acquisition reaches:

| Host | For |
|---|---|
| `api.figshare.com`, `ndownloader.figshare.com` | LUMIERE |
| `www.cancerimagingarchive.net`, `services.cancerimagingarchive.net` | UPENN-GBM, UCSD-PTGBM |
| `www.ncbi.nlm.nih.gov`, `ftp.ncbi.nlm.nih.gov` | GBM-CARE (GEO), gene annotation |
| `www.synapse.org` | GLASS / GBM-CARE consortium data (account + DUA) |

Behind a restrictive egress proxy, allow those hosts. Every download reports
its byte cost first and refuses anything above 2 GB without an explicit
`--max-bytes`.
