# Text-Based Model Loading

The PyTorch Operation Tracer now supports loading UniAD models by parsing configuration files directly, without relying on mmdet3d's registry system. This is useful for analyzing model architectures when you want more control over the model construction process.

## Usage

To use text-based model loading, add the `--text-load` flag:

```bash
python trace_ops.py --config /path/to/config.py --text-load --output trace_report.md
```

## How It Works

The text-based loader:

1. **Parses the config file**: Executes the Python config file to extract model configuration
2. **Reconstructs the model**: Manually builds the model hierarchy based on the config
3. **Handles dependencies gracefully**: Falls back to mock implementations if mmdet3d is not available

## Benefits

- **No registry dependencies**: Works without mmdet3d's global registry system
- **Clearer model construction**: See exactly how the model is built from config
- **Fallback support**: Creates mock models for tracing when dependencies are missing
- **Config variable support**: Handles configs with variables like `_dim_`, `bev_h_`, etc.

## Example

```python
# Load model using text-based parser
from utils import load_uniad_model_from_text

model, cfg = load_uniad_model_from_text(
    config_path='projects/configs/stage2_e2e/base_e2e.py',
    checkpoint_path='ckpts/uniad_base_e2e.pth',
    device='cuda'
)
```

## Mock Models

When mmdet3d is not available, the loader creates lightweight mock models that:
- Preserve the basic architecture structure
- Support forward passes for tracing
- Use simplified implementations of complex modules
- Maintain correct tensor shapes and dimensions

This allows you to analyze model structure and dataflow even without full dependencies.

## Supported Model Types

- `UniADTrack`: Base perception model for tracking and mapping
- `UniAD`: Full end-to-end model with all task heads

## Limitations

- Mock models don't implement full functionality, just structure
- Some advanced features may require mmdet3d components
- Checkpoint loading still benefits from mmdet3d's utilities

## Testing

Run the test suite to verify text-based loading:

```bash
python tests/test_text_loading.py
```