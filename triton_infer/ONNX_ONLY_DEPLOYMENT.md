# ONNX-Only Deployment Guide (No TensorRT)

## Overview
This branch (`onnx-only-optimization`) provides an optimized configuration for deploying PaddleOCR using **ONNX Runtime with CUDA Execution Provider only**, without TensorRT. This approach offers:

- ✅ **Simpler deployment** - No TensorRT engine builds
- ✅ **Predictable memory** - No engine cache, no workspace memory
- ✅ **Faster startup** - No warmup time for engine building
- ✅ **Easier debugging** - Direct ONNX execution
- ✅ **Stable performance** - No dynamic shape kernel compilation

**Trade-off:** ~20-40% slower inference than TensorRT, but still GPU-accelerated and sufficient for most use cases.

---

## Memory Budget (15GB VRAM)

### Configuration

| Component | Instances | Memory per Instance | Total Memory |
|-----------|-----------|---------------------|--------------|
| **text_detection** | 3 GPU | ~800MB | ~2.4GB |
| **text_recognition** | 4 GPU | ~500MB | ~2.0GB |
| **CUDA kernels** | - | - | ~1.5GB |
| **Dynamic buffers** | - | - | ~1.0GB |
| **Python backends** | - | - | ~0.5GB |
| **Triton overhead** | - | - | ~0.8GB |
| **TOTAL ESTIMATE** | - | - | **~8.2GB** |
| **Peak with spikes** | - | - | **~10GB** |
| **Safety margin** | - | - | **~5GB** |

### Key Differences from TensorRT

| Aspect | TensorRT Branch | ONNX-Only Branch |
|--------|----------------|------------------|
| **Memory usage** | ~12-15GB | **~8-10GB** |
| **Startup time** | 2-5 min (engine build) | **<30 sec** |
| **First inference** | Fast (pre-warmed) | **Fast (no warmup needed)** |
| **Throughput** | 100% (baseline) | ~60-80% |
| **Latency** | Lowest | +20-40% |
| **Memory spikes** | High (during build) | **Low and predictable** |
| **Deployment complexity** | High | **Low** |

---

## Configuration Details

### 1. Text Detection

**File:** `model_repository/text_detection/config.pbtxt`

```protobuf
name: "text_detection"
backend: "onnxruntime"
max_batch_size: 0

input [
  {
    name: "x"
    data_type: TYPE_FP32
    dims: [ 1, 3, -1, -1 ]
  }
]

output [
  {
    name: "fetch_name_0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, -1 ]
  }
]

instance_group [
  {
    count: 3  # Increased from 2 (no TensorRT workspace)
    kind: KIND_GPU
  }
]

# ONNX Runtime CUDA EP parameters
parameters { key: "execution_mode" value: { string_value: "1" } }
parameters { key: "intra_op_thread_count" value: { string_value: "1" } }
parameters { key: "inter_op_thread_count" value: { string_value: "1" } }
parameters { key: "arena_extend_strategy" value: { string_value: "1" } }
parameters { key: "gpu_mem_limit" value: { string_value: "4294967296" } }  # 4GB
parameters { key: "cudnn_conv_algo_search" value: { string_value: "DEFAULT" } }
parameters { key: "do_copy_in_default_stream" value: { string_value: "1" } }
parameters { key: "graph_optimization_level" value: { string_value: "2" } }
parameters { key: "enable_mem_pattern" value: { string_value: "1" } }
parameters { key: "enable_cpu_mem_arena" value: { string_value: "1" } }
```

**Key Changes:**
- ✅ **3 instances** (vs 2 with TensorRT) - more parallelism compensates for slower inference
- ✅ **4GB memory limit** (vs 5GB+2GB workspace with TensorRT)
- ✅ **Graph optimization level 2** - balances optimization vs compile time
- ✅ **No TensorRT configuration** - simpler and cleaner

### 2. Text Recognition

**File:** `model_repository/text_recognition/config.pbtxt`

```protobuf
name: "text_recognition"
backend: "onnxruntime"
max_batch_size: 0

input [
  {
    name: "x"
    data_type: TYPE_FP32
    dims: [ -1, 3, 48, -1 ]
  }
]

output [
  {
    name: "fetch_name_0"
    data_type: TYPE_FP32
    dims: [ -1, -1, 504 ]
  }
]

instance_group [
  {
    count: 4  # Increased from 2
    kind: KIND_GPU
  }
]

# ONNX Runtime CUDA EP parameters
parameters { key: "execution_mode" value: { string_value: "1" } }
parameters { key: "intra_op_thread_count" value: { string_value: "1" } }
parameters { key: "inter_op_thread_count" value: { string_value: "1" } }
parameters { key: "arena_extend_strategy" value: { string_value: "1" } }
parameters { key: "gpu_mem_limit" value: { string_value: "3221225472" } }  # 3GB
parameters { key: "cudnn_conv_algo_search" value: { string_value: "DEFAULT" } }
parameters { key: "do_copy_in_default_stream" value: { string_value: "1" } }
parameters { key: "graph_optimization_level" value: { string_value: "2" } }
parameters { key: "enable_mem_pattern" value: { string_value: "1" } }
parameters { key: "enable_cpu_mem_arena" value: { string_value: "1" } }
```

**Key Changes:**
- ✅ **4 instances** (vs 2 with TensorRT) - leverages lower memory footprint
- ✅ **3GB memory limit** (vs 4GB+1.5GB workspace with TensorRT)
- ✅ **Batching handled in preprocessing** - crops stacked before inference

---

## ONNX Runtime Parameter Reference

### Memory Management

**`arena_extend_strategy: "1"`**
- Allocates exactly what's needed (vs doubling each time)
- Critical for preventing memory spikes

**`gpu_mem_limit`**
- Hard limit on GPU memory per model
- Detection: 4GB, Recognition: 3GB
- Total: 7GB allocated, ~3GB for buffers/overhead

### Execution Settings

**`execution_mode: "1"`**
- `0` = Sequential execution
- `1` = Parallel execution (default, recommended)

**`intra_op_thread_count: "1"`**
- Threads within a single operator
- Set to 1 for GPU (CPU threads not needed)

**`inter_op_thread_count: "1"`**
- Threads for running independent operators in parallel
- Set to 1 for GPU (GPU is async already)

### CUDA EP Optimizations

**`cudnn_conv_algo_search: "DEFAULT"`**
- `EXHAUSTIVE` - Slower startup, best performance
- `DEFAULT` - **Balance speed/memory (recommended)**
- `HEURISTIC` - Fastest startup, good performance

**`do_copy_in_default_stream: "1"`**
- Improves multi-instance performance
- Reduces stream synchronization overhead

**`graph_optimization_level: "2"`**
- `0` = Disabled
- `1` = Basic (conv+bn fusion, etc.)
- `2` = **Extended (recommended for CUDA EP)**
- `3` = All (includes layout transformations)

Level 2 is optimal - level 3 can sometimes hurt CUDA performance.

### Memory Pattern

**`enable_mem_pattern: "1"`**
- Enables memory pattern optimization
- Pre-allocates based on observed patterns
- Reduces fragmentation

**`enable_cpu_mem_arena: "1"`**
- Enables CPU memory arena
- Even GPU models use some CPU memory
- Reduces malloc overhead

---

## Performance Expectations

### Throughput Comparison

| Scenario | TensorRT | ONNX-Only | Delta |
|----------|----------|-----------|-------|
| **Single 960x960 image** | 100ms | 140ms | +40% |
| **Batch 10 text crops** | 30ms | 45ms | +50% |
| **Sustained throughput** | 15 img/s | 10 img/s | -33% |
| **Memory usage** | 12-15GB | 8-10GB | **-30%** |
| **Startup time** | 3 min | 20 sec | **-87%** |

### When ONNX-Only is Better

✅ **Development & Testing**
- Faster iteration cycles
- Easier debugging
- No engine rebuild on config changes

✅ **Memory-Constrained Environments**
- Shared GPU with other services
- Smaller GPUs (<16GB VRAM)
- Cost-sensitive deployments

✅ **Variable Input Sizes**
- Wide range of image sizes
- No penalty for new shapes
- Consistent latency

✅ **Simple Deployment**
- No TensorRT version management
- Works across CUDA versions
- Fewer dependencies

### When TensorRT is Better

✅ **Production with fixed workloads**
- Consistent input sizes
- Maximum throughput needed
- Dedicated GPU

✅ **High-volume processing**
- >20 images/second
- Batch processing
- Ultra-low latency requirements

✅ **Cost per inference**
- Amortize startup cost
- Long-running services
- High GPU utilization

---

## Deployment Instructions

### 1. Switch to ONNX-Only Branch

```bash
cd /path/to/triton_infer
git checkout onnx-only-optimization
```

### 2. Launch Triton Server

```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/workspace/models/model_repository \
  --name triton-ocr \
  nvcr.io/nvidia/tritonserver:24.12-py3 \
  tritonserver \
    --model-repository=/workspace/models/model_repository \
    --backend-config=onnxruntime,memory_limit_mb=10240 \
    --log-verbose=1 \
    --strict-model-config=false
```

### 3. Verify Models Loaded

```bash
curl localhost:8000/v2/health/ready
curl localhost:8000/v2/models/ensemble_model
```

### 4. Monitor Memory Usage

```bash
nvidia-smi dmon -s mu -c 100
```

---

## Tuning Guide

### If Memory is Still Too High

1. **Reduce instance counts:**
   ```
   text_detection: count: 2
   text_recognition: count: 3
   ```

2. **Lower memory limits:**
   ```
   text_detection: gpu_mem_limit: "3221225472"  # 3GB
   text_recognition: gpu_mem_limit: "2147483648"  # 2GB
   ```

3. **Disable memory pattern:**
   ```
   parameters { key: "enable_mem_pattern" value: { string_value: "0" } }
   ```

### If Performance is Too Slow

1. **Increase instance counts:**
   ```
   text_detection: count: 4
   text_recognition: count: 5
   ```

2. **Try EXHAUSTIVE algo search** (one-time cost):
   ```
   parameters { key: "cudnn_conv_algo_search" value: { string_value: "EXHAUSTIVE" } }
   ```

3. **Enable graph optimization level 3:**
   ```
   parameters { key: "graph_optimization_level" value: { string_value: "3" } }
   ```

   **Warning:** Level 3 can sometimes hurt CUDA performance. Benchmark first!

4. **Add warmup** (optional, minimal benefit for CUDA EP):
   ```protobuf
   model_warmup [
     {
       name: "warm_960"
       count: 1
       inputs {
         key: "x"
         value: {
           data_type: TYPE_FP32
           dims: [ 1, 3, 960, 960 ]
           zero_data: true
         }
       }
     }
   ]
   ```

### If Memory Spikes Still Occur

1. **Enable deterministic compute:**
   ```
   parameters { key: "use_deterministic_compute" value: { string_value: "1" } }
   ```

   This can reduce memory spikes at cost of slight performance loss.

2. **Reduce graph optimization level:**
   ```
   parameters { key: "graph_optimization_level" value: { string_value: "1" } }
   ```

3. **Check for memory leaks:**
   ```bash
   nvidia-smi --query-gpu=memory.used --format=csv -l 1
   ```

---

## Monitoring & Validation

### Check Execution Provider

```bash
# Check Triton logs to confirm CUDA EP is used
docker logs triton-ocr 2>&1 | grep -i "execution provider"

# Should see: "Adding default CPU execution provider" and CUDA kernels
# Should NOT see: "TensorRT EP"
```

### Benchmark Performance

```python
import tritonclient.http as httpclient
import numpy as np
import time

client = httpclient.InferenceServerClient(url="localhost:8000")

# Test image
img = np.random.randint(0, 255, (960, 960, 3), dtype=np.uint8)

# Warm up
for _ in range(5):
    # ... make inference ...
    pass

# Benchmark
times = []
for _ in range(50):
    start = time.time()
    # ... make inference ...
    times.append(time.time() - start)

print(f"Mean latency: {np.mean(times)*1000:.1f}ms")
print(f"P95 latency: {np.percentile(times, 95)*1000:.1f}ms")
print(f"Throughput: {1/np.mean(times):.1f} img/s")
```

### Compare Memory Usage

```bash
# ONNX-only branch
nvidia-smi

# Should see ~8-10GB used

# vs TensorRT branch (for comparison)
# Would see ~12-15GB used
```

---

## Troubleshooting

### Models not loading

**Check logs:**
```bash
docker logs triton-ocr 2>&1 | grep -i error
```

**Common issues:**
- Missing ONNX Runtime CUDA EP (check Triton image version)
- GPU memory exhausted (reduce instance counts)
- Invalid parameter names (check spelling)

### Slower than expected

**Verify CUDA EP is being used:**
```bash
docker logs triton-ocr 2>&1 | grep -i "cuda"
```

**Check GPU utilization:**
```bash
nvidia-smi dmon -s u -c 10
```

Low utilization? Increase instance counts.

**Profile inference:**
```bash
# Enable profiling in config
parameters { key: "enable_profiling" value: { string_value: "1" } }
```

### Memory still growing

**Check for leaks:**
```bash
# Monitor over time
watch -n 1 nvidia-smi

# If memory grows continuously, check:
# 1. Python backend memory leaks (preprocessing/postprocessing)
# 2. Client-side issues (not releasing responses)
# 3. Triton response cache (disable if needed)
```

---

## Migration Between Branches

### From TensorRT to ONNX-Only

```bash
git checkout onnx-only-optimization

# Clear TensorRT cache (no longer needed)
rm -rf model_repository/_trt_cache/

# Restart Triton
docker restart triton-ocr
```

**No code changes needed** - only configs differ.

### From ONNX-Only to TensorRT

```bash
git checkout trt

# First startup will build TensorRT engines (2-5 min)
docker restart triton-ocr

# Monitor startup
docker logs -f triton-ocr
```

---

## Summary

This ONNX-only configuration provides:

| Metric | Value |
|--------|-------|
| **VRAM usage** | ~8-10GB (vs ~12-15GB with TensorRT) |
| **Startup time** | <30 seconds (vs 2-5 min with TensorRT) |
| **Throughput** | ~60-80% of TensorRT |
| **Latency** | +20-40% vs TensorRT |
| **Deployment complexity** | Low |
| **Memory predictability** | High |
| **Production ready** | ✅ Yes (for moderate workloads) |

**Best for:** Development, testing, memory-constrained environments, simple deployment

**Consider TensorRT for:** High-throughput production, dedicated GPUs, maximum performance

---

## Additional Resources

- [ONNX Runtime CUDA EP Docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/tune-performance.html)
- [Triton ONNX Backend](https://github.com/triton-inference-server/onnxruntime_backend)

---

**Branch:** `onnx-only-optimization`
**Deployment target:** 15GB VRAM budget, shared GPU environment
**Performance target:** Moderate throughput (5-15 images/second)
