# TensorRT Optimization Guide for PaddleOCR Triton

## Overview
This repository has been optimized for TensorRT serving to minimize kernel warmup time and maximize OCR throughput. The key changes address dynamic input shape handling and efficient batching.

## Key Optimizations

### 1. Text Detection Model (`text_detection`)

**Problem:** Different image resolutions caused constant TensorRT kernel warmups.

**Solution:**
- Enabled TensorRT execution with FP16 precision
- Configured optimization profiles covering common shape ranges:
  - Min: 3x320x320
  - Optimal: 3x960x960
  - Max: 3x1920x1920
- Added warmup for common sizes (640, 960, 1280) to pre-build engines
- Preprocessing already rounds to 32-pixel multiples for optimal performance

**Config:** `model_repository/text_detection/config.pbtxt`

### 2. Text Recognition Model (`text_recognition`)

**Problem:**
- Variable-width text crops caused kernel warmups for each unique width
- Processing crops individually was inefficient

**Solution:**
- **Enabled dynamic batching** (max_batch_size: 64) to batch multiple text crops together
- Configured preferred batch sizes: [4, 8, 16, 32, 64] for optimal GPU utilization
- Set max queue delay to 5ms for low latency while allowing batching
- Optimization profiles for width dimension (height fixed at 48):
  - Min width: 32
  - Optimal width: 320
  - Max width: 960
- Multiple warmup configurations for common batch sizes and widths
- **Quantized widths to multiples of 32** in preprocessing to increase TensorRT cache hits

**Config:** `model_repository/text_recognition/config.pbtxt`

### 3. Preprocessing Improvements

**Recognition Preprocessing** (`rec_preprocess.py`):
- Groups crops by similar aspect ratios (rounded to nearest 0.5)
- Quantizes output widths to multiples of 32 pixels
- Reduces unique input shape combinations for better TensorRT kernel reuse
- Clamps widths to reasonable range (32-960) matching optimization profiles

**Detection Preprocessing** (`det_preprocess.py`):
- Already rounds to 32-pixel multiples
- Optional size quantization available (commented out) for even more aggressive caching
- Consistent sizing reduces TensorRT engine rebuild frequency

## Performance Benefits

### Before Optimization
- ❌ New kernel warmup for each unique input size
- ❌ Text crops processed individually (no batching)
- ❌ Unpredictable latency spikes on new shapes
- ❌ Suboptimal GPU utilization

### After Optimization
- ✅ Pre-built TensorRT engines for common shapes
- ✅ Dynamic batching of text crops (up to 64x)
- ✅ Quantized shapes maximize kernel cache hits
- ✅ 5ms max queue delay for near-real-time batching
- ✅ Reduced instance count for recognition (6→4) due to batching efficiency
- ✅ FP16 precision for 2x speedup vs FP32

## Configuration Details

### TensorRT Cache
Engines are cached in Docker at:
- Detection: `/workspace/models/_trt_cache/text_detection`
- Recognition: `/workspace/models/_trt_cache/text_recognition`

Ensure these directories are persisted across container restarts.

### Dynamic Batching Parameters
```
preferred_batch_size: [4, 8, 16, 32, 64]
max_queue_delay_microseconds: 5000
```

Triton will:
1. Wait up to 5ms to accumulate requests
2. Form batches at preferred sizes when possible
3. Submit batch to GPU for efficient parallel processing

### Optimization Profiles
Profiles define shape ranges TensorRT optimizes for:
- **Min shapes:** Smallest expected input (must be >= actual minimum)
- **Opt shapes:** Most common input size (optimized most heavily)
- **Max shapes:** Largest expected input (must be >= actual maximum)

Inputs outside these ranges will trigger engine rebuilds.

## Monitoring & Tuning

### Check TensorRT Engine Status
```bash
# Check if engines are being built
docker logs <triton-container> | grep -i "tensorrt"

# Monitor cache directory
ls -lh /workspace/models/_trt_cache/*/
```

### Tuning Recommendations

**If you see frequent kernel warmups:**
1. Check input size distribution in your workload
2. Adjust optimization profiles to better match your data
3. Enable size quantization in `det_preprocess.py` (commented out)
4. Add more warmup shapes matching your common inputs

**If latency is too high:**
1. Reduce `max_queue_delay_microseconds` (trade batching for latency)
2. Reduce `preferred_batch_size` values
3. Increase instance count if batches are large

**If throughput is low:**
1. Increase `max_queue_delay_microseconds` (allow more batching)
2. Increase `max_batch_size` if you have GPU memory
3. Monitor GPU utilization and adjust instance counts

### Size Distribution Analysis
Add logging to preprocessing to understand your shape distribution:
```python
# In det_preprocess.py or rec_preprocess.py
print(f"Shape: {resize_h}x{resize_w}")
```

Adjust profiles and warmup based on the most common sizes.

## Implementation Notes

### Shape Consistency
- Detection input: Dynamic (3, H, W) where H,W are multiples of 32
- Recognition input: Dynamic (N, 3, 48, W) where W is multiple of 32
- Batching only on recognition (detection processes 1 image at a time)

### Bounding Box Calculation
Shape information is properly transmitted through the pipeline via `shape_list` tensor, ensuring bounding boxes are correctly calculated regardless of preprocessing resizing.

### Width Grouping
Recognition preprocessing groups crops by similar aspect ratios before processing. This ensures:
1. Crops with similar widths are padded to same dimensions
2. Triton's dynamic batcher can group them efficiently
3. Single GPU kernel processes entire batch
4. Reduced kernel launch overhead

## Troubleshooting

### "TensorRT engine build failed"
- Check GPU memory (increase `max_workspace_size_bytes` or reduce other loads)
- Verify input shapes are within optimization profile ranges
- Check Triton logs for detailed error messages

### "Shape mismatch" errors
- Ensure preprocessing outputs match config.pbtxt input shapes
- Verify batch dimension handling (0 vs -1 in dims)
- Check that dynamic axes are correctly specified

### Slow first inference
- Verify warmup is configured and executing at startup
- Check TensorRT cache directory is writable and persisted
- First run after code changes will rebuild engines (expected)

### Recognition accuracy issues
- Width quantization shouldn't affect accuracy (still preserves aspect ratio)
- If accuracy degrades, reduce quantization aggressiveness
- Verify normalization (mean/std) is unchanged

## Further Optimizations

Consider these additional improvements:

1. **CUDA Graphs** - Currently disabled, but can reduce kernel launch overhead
   - Enable with `trt_cuda_graph_enable: "1"` after verifying shape consistency

2. **INT8 Quantization** - Further speedup with calibration
   - Requires representative calibration dataset
   - May slightly reduce accuracy

3. **Model Distillation** - Smaller models are faster
   - Consider mobile variants of PaddleOCR models

4. **Async Execution** - Pipeline parallelism
   - Detection and recognition from different images can overlap

5. **Multi-Stream** - Handle multiple requests in parallel
   - Already supported by instance_group configuration

## Testing

After deployment, test with varying input sizes:
```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

# Test different sizes
for size in [640, 960, 1280]:
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    # ... send to ensemble_model ...

# Check that subsequent calls are fast (no warmup)
```

Monitor Triton metrics:
```bash
curl localhost:8002/metrics | grep -E "(queue|batch|compute)"
```

## Summary

The optimization strategy focuses on:
1. **Predictable shapes** - Quantization reduces unique combinations
2. **Efficient batching** - Dynamic batching maximizes GPU utilization
3. **Pre-built engines** - Warmup eliminates first-inference penalties
4. **Proper profiles** - Shape ranges cover expected inputs without excess

This results in consistent, low-latency OCR performance with high throughput.
