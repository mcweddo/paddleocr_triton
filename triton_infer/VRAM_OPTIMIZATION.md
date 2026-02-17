# VRAM Optimization Guide - 15GB Memory Budget

## Overview
This configuration optimizes the PaddleOCR Triton deployment for **15GB VRAM maximum usage**, suitable for shared GPU environments where other model servers coexist.

## Memory Budget Breakdown

### Optimized Configuration (~10-12GB total)

| Component | Instances | Memory per Instance | Total Memory |
|-----------|-----------|---------------------|--------------|
| **text_detection** | 2 GPU | ~250MB + engines | ~2.5GB |
| **text_recognition** | 2 GPU | ~150MB + engines | ~2.0GB |
| **TensorRT Workspaces** | - | - | ~3.5GB |
| **Detection Engines** | - | 2GB workspace | ~2GB |
| **Recognition Engines** | - | 1.5GB workspace | ~1.5GB |
| **Dynamic Batching Buffers** | - | - | ~1.5GB |
| **Python Backends (CPU)** | - | Minimal | ~0.5GB |
| **Response Cache** | - | 256MB | ~0.25GB |
| **Triton Overhead** | - | - | ~1GB |
| **TOTAL ESTIMATE** | - | - | **~10-12GB** |

**Safety margin:** ~3-5GB for memory spikes and other services

## Changes from Default Configuration

### 1. Reduced GPU Instance Counts

**text_detection:**
- Before: 3 instances
- After: **2 instances**
- Reason: Detection is relatively fast; 2 instances handle typical load

**text_recognition:**
- Before: 4 instances (was 6 originally)
- After: **2 instances**
- Reason: Dynamic batching compensates for fewer instances

### 2. Reduced TensorRT Workspace Sizes

**text_detection:**
- Before: 4GB (`4294967296` bytes)
- After: **2GB** (`2147483648` bytes)
- Impact: May slightly reduce optimization opportunities for very large shapes

**text_recognition:**
- Before: 4GB
- After: **1.5GB** (`1610612736` bytes)
- Impact: Minimal, as recognition model is smaller

### 3. Optimized Dynamic Batching

**text_recognition:**
- Preferred batch sizes: `[8, 16, 32]` (removed 4 and 64)
- Max queue delay: **8ms** (increased from 5ms)
- Reason: Larger batches compensate for fewer instances

### 4. Reduced CPU Instances

**detection_preprocessing:**
- Before: 2 CPU instances
- After: **1 CPU instance**

**detection_postprocessing:**
- Before: 2 CPU instances
- After: **1 CPU instance**

**recognition_postprocessing:**
- Before: 4 CPU instances
- After: **2 CPU instances**

### 5. Reduced Warmup Iterations

**text_detection:**
- Before: 4 warmup runs (2 + 1 + 1)
- After: **2 warmup runs** (960x960, 640x640)

**text_recognition:**
- Before: 5 warmup runs
- After: **2 warmup runs** (batch 16@320, batch 8@160)

### 6. Rate Limiting

Added global rate limiter to prevent memory spikes:
```
rate_limiter {
  resources [
    {
      name: "gpu_memory"
      global: true
      count: 2
    }
  ]
}
```

Limits to 2 concurrent pipeline executions through ensemble.

## Server Launch Configuration

### Recommended Launch Command

```bash
tritonserver \
  --model-repository=/workspace/models/model_repository \
  --model-control-mode=explicit \
  --load-model=ensemble_model \
  --strict-model-config=false \
  --backend-config=onnxruntime,memory_limit_mb=12288 \
  --backend-config=python,shm-default-byte-size=1048576000 \
  --response-cache-byte-size=268435456 \
  --log-verbose=1 \
  --exit-timeout-secs=120
```

### Launch Parameters Explained

- `--model-control-mode=explicit`: Only load specified models (saves memory)
- `--load-model=ensemble_model`: Load ensemble and dependencies only
- `--backend-config=onnxruntime,memory_limit_mb=12288`: Hard limit 12GB for ONNX/TensorRT
- `--backend-config=python,shm-default-byte-size=1048576000`: 1GB shared memory for Python
- `--response-cache-byte-size=268435456`: 256MB response cache

### Alternative: Use Config File

```bash
tritonserver \
  --model-repository=/workspace/models/model_repository \
  --model-config-path=/workspace/models/config.pbtxt \
  --log-verbose=1
```

The `config.pbtxt` file contains all optimizations.

## Monitoring VRAM Usage

### Check Current Usage

```bash
# Inside container or on host
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 1

# Or use Triton metrics
curl localhost:8002/metrics | grep -E "(gpu_memory|gpu_utilization)"
```

### Monitor Model Memory

```bash
# Triton model statistics
curl localhost:8000/v2/models/stats | jq

# Look for:
# - model_inference_count
# - model_inference_duration
# - queue_time
```

### Check TensorRT Engine Sizes

```bash
# Inside container
du -sh /workspace/models/_trt_cache/*/
ls -lh /workspace/models/_trt_cache/*/*.engine
```

## Performance Impact Analysis

### Throughput Changes

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Detection instances | 3 | 2 | -33% capacity |
| Recognition instances | 4 | 2 | -50% capacity |
| Recognition batch size | 64 | 32 | -50% max batch |
| Batching delay | 5ms | 8ms | +60% accumulation |
| **Net throughput** | 100% | **~70-80%** | -20-30% |

### Latency Changes

- **Single image:** +2-3ms (longer queue delay)
- **Burst traffic:** Similar (batching compensates)
- **Sustained load:** +10-15% (fewer instances)

### When This Configuration Works Best

✅ **Good for:**
- Moderate request rates (<20 req/s)
- Bursty traffic patterns (batching helps)
- Mixed workloads with other GPU services
- Cost-sensitive deployments

❌ **Not ideal for:**
- Very high sustained throughput (>50 req/s)
- Ultra-low latency requirements (<20ms)
- Dedicated GPU environments

## Tuning for Your Workload

### If VRAM Usage is Still Too High

1. **Further reduce instances:**
   ```
   text_detection: count: 1
   text_recognition: count: 1
   ```

2. **Reduce TensorRT workspaces:**
   ```
   text_detection: max_workspace_size_bytes: 1073741824  # 1GB
   text_recognition: max_workspace_size_bytes: 805306368  # 768MB
   ```

3. **Disable warmup temporarily:**
   - Comment out `model_warmup` sections
   - First inference will be slower, but saves startup memory

4. **Use model control mode with lazy loading:**
   ```bash
   --model-control-mode=explicit
   --load-model=ensemble_model  # Only load when needed
   ```

### If Throughput is Too Low

1. **Increase queue delay:**
   ```
   max_queue_delay_microseconds: 10000  # 10ms
   ```

2. **Re-enable larger batch sizes:**
   ```
   preferred_batch_size: [ 8, 16, 32, 48 ]
   ```

3. **If you have 1-2GB more VRAM:**
   ```
   text_recognition: count: 3
   ```

### If Latency is Too High

1. **Reduce queue delay:**
   ```
   max_queue_delay_microseconds: 3000  # 3ms
   ```

2. **Remove rate limiter:**
   - Comment out `rate_limiter` in ensemble config
   - Risk: memory spikes under load

3. **Prioritize single-image performance:**
   ```
   preferred_batch_size: [ 4, 8, 16 ]  # Favor smaller batches
   ```

## Load Testing

Test your memory-optimized configuration:

```python
import tritonclient.http as httpclient
import numpy as np
import concurrent.futures
import time

def send_request(client, img_id):
    img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    # ... setup triton request ...
    start = time.time()
    result = client.infer("ensemble_model", inputs=[...])
    latency = time.time() - start
    return img_id, latency

client = httpclient.InferenceServerClient(url="localhost:8000")

# Concurrent load test
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(send_request, client, i) for i in range(100)]
    results = [f.result() for f in futures]

# Analyze results
latencies = [r[1] for r in results]
print(f"Mean latency: {np.mean(latencies):.3f}s")
print(f"P95 latency: {np.percentile(latencies, 95):.3f}s")
print(f"P99 latency: {np.percentile(latencies, 99):.3f}s")
```

While running, monitor VRAM:
```bash
watch -n 0.5 nvidia-smi
```

## Troubleshooting

### "CUDA out of memory" errors

1. **Check actual usage:**
   ```bash
   nvidia-smi dmon -s mu -c 100
   ```

2. **Further reduce workspaces:**
   - text_detection: 1GB
   - text_recognition: 512MB

3. **Disable response cache:**
   ```bash
   --response-cache-byte-size=0
   ```

4. **Check for memory leaks:**
   - Restart Triton periodically
   - Monitor over time: `nvidia-smi --query-gpu=memory.used --format=csv -l 60`

### Slower than expected

1. **Check if engines are built:**
   ```bash
   ls -la /workspace/models/_trt_cache/
   ```

2. **Verify batching is working:**
   ```bash
   curl localhost:8002/metrics | grep batch
   ```

3. **Check CPU bottlenecks:**
   - Increase preprocessing instances if CPU-bound

### Models not loading

1. **Check dependencies:**
   ```bash
   curl localhost:8000/v2/models/text_detection
   curl localhost:8000/v2/models/text_recognition
   ```

2. **Review logs:**
   ```bash
   docker logs <container> 2>&1 | grep -i error
   ```

3. **Verify model files exist:**
   ```bash
   ls -la model_repository/*/1/model.*
   ```

## Comparison with Standard Config

| Aspect | Standard (TENSORRT_OPTIMIZATION.md) | Memory-Optimized (This Guide) |
|--------|-------------------------------------|-------------------------------|
| VRAM Usage | ~18-20GB | **~10-12GB** |
| Throughput | 100% | ~70-80% |
| Latency (avg) | Baseline | +10-15% |
| Max Batch | 64 | 32 |
| Instances (total) | 7 GPU + 8 CPU | **4 GPU + 4 CPU** |
| TensorRT Workspace | 8GB total | **3.5GB total** |
| Queue Delay | 5ms | 8ms |
| Concurrent Limit | Unlimited | 2 pipelines |

## Summary

This configuration reduces VRAM from ~18GB to **~10-12GB** by:

1. ✅ Halving GPU instance counts (7 → 4)
2. ✅ Reducing TensorRT workspaces (8GB → 3.5GB)
3. ✅ Optimizing dynamic batching parameters
4. ✅ Reducing CPU instances
5. ✅ Adding rate limiting
6. ✅ Minimal warmup iterations

**Trade-off:** ~20-30% throughput reduction for ~40% memory savings.

This is ideal for **shared GPU environments** where multiple services need VRAM and moderate OCR throughput is acceptable.
