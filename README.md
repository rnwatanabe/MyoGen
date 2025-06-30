# MyoGen

## How to install

### Clone the repository
```bash
git clone https://github.com/NsquaredLab/MyoGen.git
```

### (Windows only) Install NEURON 8.2.6
Install NEURON 8.2.6 from https://github.com/neuronsimulator/nrn/releases/download/8.2.6/nrn-8.2.6.w64-mingw-py-38-39-310-311-312-setup.exe

### Install UV
https://docs.astral.sh/uv/#highlights

### Create environment
```bash
uv sync
```

### Add cupy if CUDA is available
```bash
uv pip install cupy-cuda12x
```

### Activate the environment
```bash
source .venv/bin/activate
```

### Run the setup to compile the NMODL files correctly
```bash
python -c "from myogen.utils import setup_myogen; setup_myogen()"
```

## NMODL/NEURON Issues

If you encounter NMODL compilation or loading issues:

1. **Automatic handling**: MyoGen automatically handles NMODL loading. Just import and use:
   ```python
   from myogen import simulator  # NMODL files loaded automatically
   ```

2. **Manual setup** (if needed):
   ```python
   from myogen.utils import setup_myogen
   setup_myogen()  # Explicit setup with verbose output
   from myogen import simulator
   ```

3. **Force recompilation**:
   ```python
   from myogen.utils import setup_myogen
   setup_myogen(force_nmodl_reload=True)  # Force recompilation
   ```

## Examples

See the `examples/` directory for detailed usage examples.

