# Model Converter

A collection of utilities for working with safetensors model files, including dtype conversion and inspection tools.

## Features

- **Convert between bf16 and fp16**: Convert safetensors model files between bfloat16 and float16 precision
- **Inspect tensor dtypes**: Check the data types of tensors in safetensors files
- **Batch conversion**: Convert single files, sharded models with index.json, or entire directories
- **Preserve metadata**: Maintains model metadata during conversion with conversion history

## Installation

1. Clone the repository:
```bash
git clone git@github.com:joewhaley/model_converter.git
cd model_converter
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- safetensors >= 0.3.0
- numpy >= 1.24.0
- packaging >= 23.0

## Usage

### Print Model Tensor Types

Inspect the data types of tensors in a safetensors file:

```bash
python print_model_type.py <path-to-model.safetensors>
```

**Example:**
```bash
python print_model_type.py model.safetensors
# Output: {'torch.bfloat16': 120, 'torch.int64': 5}
```

This tool counts how many tensors of each dtype are present in the file.

### Convert Safetensors Dtype

Convert model files between bfloat16 (bf16) and float16 (fp16) precision.

**Basic Usage:**
```bash
python convert_safetensors_dtype.py --in <input> --out <output> --to <bf16|fp16>
```

**Examples:**

1. **Convert a single file:**
```bash
python convert_safetensors_dtype.py --in model.safetensors --out model_fp16.safetensors --to fp16
```

2. **Convert a sharded model with index.json:**
```bash
python convert_safetensors_dtype.py --in model.safetensors.index.json --out output_dir/ --to bf16
```

3. **Convert all files in a directory:**
```bash
python convert_safetensors_dtype.py --in input_models/ --out output_models/ --to fp16
```

**Notes:**
- Only floating-point tensors are converted; integer and boolean tensors remain unchanged
- Conversion runs on CPU by default for compatibility
- The tool adds metadata to track conversion history
- Sharded models maintain their structure with updated index files

## How It Works

### Dtype Conversion

The converter:
1. Loads safetensors files into CPU memory
2. Identifies floating-point tensors
3. Converts them to the target dtype (bf16 or fp16)
4. Preserves non-floating-point tensors as-is
5. Updates metadata with conversion information
6. Saves the converted model

### Supported Formats

- **Single files**: Individual `.safetensors` files
- **Sharded models**: Models split across multiple `.safetensors` files with an `index.json` or `model.safetensors.index.json` file
- **Directories**: Batch process all safetensors files in a folder

## Performance Considerations

- Conversion is performed on CPU by default for maximum compatibility
- For faster conversion on CUDA systems, you could modify the code to use GPU
- Memory usage depends on model size; ensure sufficient RAM is available

## License

MIT License - feel free to use and modify as needed.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

