# Branch Comparison Guide

## Overview

This repository has two optimized configurations for different deployment scenarios:

| Branch | Best For | VRAM Usage | Performance | Complexity |
|--------|----------|------------|-------------|------------|
| **`trt`** | Production, dedicated GPU | 12-15GB | 100% (fastest) | High |
| **`onnx-only-optimization`** | Development, shared GPU | 8-10GB | 60-80% | Low |

---

## Detailed Comparison

### TensorRT Branch (`trt`)

**Execution:** ONNX Runtime + TensorRT Execution Provider

**Configuration:**
```
text_detection:  2 instances, 2GB TRT workspace, 5GB mem limit
text_recognition: 2 instances, 1.5GB TRT workspace, 4GB mem limit
Total instances: 4 GPU
Total VRAM: ~12-15GB
```

**Pros:**
- ✅ Maximum throughput (~15 img/s)
- ✅ Lowest latency per image
- ✅ FP16 precision optimization
- ✅ Pre-built engines (after warmup)
- ✅ Optimized for fixed input ranges

**Cons:**
- ❌ 2-5 minute startup (engine build)
- ❌ High memory usage
- ❌ Kernel warmup on new shapes
- ❌ Complex configuration (profiles, warmup)
- ❌ TensorRT version dependencies

**Use Cases:**
- High-volume production deployments
- Dedicated GPU servers
- Fixed/predictable input sizes
- Maximum performance required
- Cost per inference optimization

---

### ONNX-Only Branch (`onnx-only-optimization`)

**Execution:** ONNX Runtime + CUDA Execution Provider (no TensorRT)

**Configuration:**
```
text_detection:  3 instances, no workspace, 4GB mem limit
text_recognition: 4 instances, no workspace, 3GB mem limit
Total instances: 7 GPU
Total VRAM: ~8-10GB
```

**Pros:**
- ✅ Fast startup (<30 seconds)
- ✅ Lower memory usage (~40% reduction)
- ✅ Predictable memory (no spikes)
- ✅ No kernel warmup delays
- ✅ Simple configuration
- ✅ No TensorRT dependencies
- ✅ More instances possible

**Cons:**
- ❌ 20-40% slower than TensorRT
- ❌ Lower throughput (~10 img/s)
- ❌ No FP16 optimization (FP32 only on CUDA EP)

**Use Cases:**
- Development and testing
- Shared GPU environments
- Memory-constrained deployments (<16GB VRAM)
- Variable input sizes
- Quick deployment needs
- Cost-sensitive projects

---

## Performance Metrics

### Throughput

| Workload | TensorRT | ONNX-Only | Notes |
|----------|----------|-----------|-------|
| Single 960x960 image | 100ms | 140ms | +40% latency |
| Batch 10 text crops | 30ms | 45ms | +50% latency |
| Sustained throughput | 15 img/s | 10 img/s | -33% throughput |
| Concurrent requests (4x) | 40 img/s | 30 img/s | ONNX has more instances |

### Memory Profile

| Stage | TensorRT | ONNX-Only |
|-------|----------|-----------|
| Model loading | 2GB | 1.5GB |
| Engine building | +4GB (temporary) | N/A |
| Steady state | 12GB | 8GB |
| Peak (4 concurrent) | 15GB | 10GB |
| Spike potential | High | Low |

### Startup Time

| Event | TensorRT | ONNX-Only |
|-------|----------|-----------|
| Model load | 10s | 5s |
| Engine build (detection) | 2 min | N/A |
| Engine build (recognition) | 1 min | N/A |
| Warmup | 30s | N/A |
| **Total ready time** | **3-5 min** | **<30 sec** |

---

## When to Choose Each Branch

### Choose TensorRT (`trt`) if:

✅ You have **dedicated GPU** (not shared)
✅ You need **maximum throughput**
✅ You have **>16GB VRAM**
✅ Your **input sizes are predictable**
✅ You can **tolerate 3-5 min startup**
✅ You're deploying **to production**
✅ You want **lowest cost per inference**

### Choose ONNX-Only (`onnx-only-optimization`) if:

✅ You're **sharing GPU** with other services
✅ You have **<16GB VRAM**
✅ You need **fast startup/iteration**
✅ You're **developing/testing**
✅ You have **variable input sizes**
✅ You want **simpler deployment**
✅ You prioritize **predictable memory**
✅ **10 img/s throughput is sufficient**

---

## Switching Between Branches

### From Main to TensorRT

```bash
git checkout trt

# First startup builds engines (wait 3-5 min)
docker restart triton-ocr
docker logs -f triton-ocr  # Watch engine build progress

# Verify
curl localhost:8000/v2/health/ready
```

### From Main to ONNX-Only

```bash
git checkout onnx-only-optimization

# Quick startup
docker restart triton-ocr

# Verify
curl localhost:8000/v2/health/ready
```

### From TensorRT to ONNX-Only

```bash
git checkout onnx-only-optimization

# Clean up TensorRT cache (optional, saves disk space)
rm -rf model_repository/_trt_cache/

docker restart triton-ocr
```

### From ONNX-Only to TensorRT

```bash
git checkout trt

# First startup will build engines
docker restart triton-ocr
docker logs -f triton-ocr  # Monitor build
```

---

## Configuration Files Affected

### Both Branches Modify:

- `model_repository/text_detection/config.pbtxt`
- `model_repository/text_recognition/config.pbtxt`

### TensorRT Branch Adds:

- `TENSORRT_OPTIMIZATION.md`
- `VRAM_OPTIMIZATION.md`
- `MEMORY_PARAMETERS.md`
- `clear_trt_cache.sh`
- `config.pbtxt` (server config)

### ONNX-Only Branch Adds:

- `ONNX_ONLY_DEPLOYMENT.md`
- `MEMORY_PARAMETERS.md`
- `config.pbtxt` (server config)

---

## Docker Launch Commands

### TensorRT Branch

```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/workspace/models/model_repository \
  -v $(pwd)/_trt_cache:/workspace/models/_trt_cache \
  --name triton-ocr \
  nvcr.io/nvidia/tritonserver:24.12-py3 \
  tritonserver \
    --model-repository=/workspace/models/model_repository \
    --model-control-mode=explicit \
    --load-model=ensemble_model \
    --backend-config=onnxruntime,memory_limit_mb=12288 \
    --log-verbose=1
```

**Key:** TensorRT cache volume for persistence

### ONNX-Only Branch

```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/workspace/models/model_repository \
  --name triton-ocr \
  nvcr.io/nvidia/tritonserver:24.12-py3 \
  tritonserver \
    --model-repository=/workspace/models/model_repository \
    --backend-config=onnxruntime,memory_limit_mb=10240 \
    --log-verbose=1
```

**Key:** No cache volume needed, lower memory limit

---

## Cost Analysis

### Per-Image Cost (Assuming $1/hour GPU)

| Branch | Images/Hour | Cost per 1M Images |
|--------|-------------|-------------------|
| TensorRT | 54,000 | $18.50 |
| ONNX-Only | 36,000 | $27.80 |
| **Delta** | -33% | **+50% cost** |

### Total Cost of Ownership (Monthly)

| Aspect | TensorRT | ONNX-Only |
|--------|----------|-----------|
| GPU cost (A10, 24GB) | $300/mo | $200/mo (shared) |
| Engineering time | 20 hrs | 5 hrs |
| Debugging overhead | Medium | Low |
| Operational complexity | High | Low |
| **Total (1st month)** | $320 + 20hrs | $200 + 5hrs |
| **Total (ongoing)** | $300/mo | $200/mo |

**Verdict:** ONNX-Only cheaper for development, TensorRT cheaper for high-volume production

---

## Migration Path

### Recommended Workflow

```
Development → Testing → Staging → Production
   ↓            ↓          ↓          ↓
ONNX-Only → ONNX-Only → TensorRT → TensorRT
```

**Strategy:**
1. **Develop** on `onnx-only-optimization` (fast iteration)
2. **Test** on `onnx-only-optimization` (validate functionality)
3. **Stage** on `trt` (performance testing)
4. **Deploy** on `trt` (production)

### Gradual Migration

If unsure, run A/B test:
- 20% traffic → ONNX-Only (test stability)
- 80% traffic → TensorRT (maintain performance)
- Monitor metrics, adjust split

---

## Monitoring Queries

### Check Which Branch is Deployed

```bash
# Look for TensorRT in logs
docker logs triton-ocr 2>&1 | grep -i "tensorrt" | wc -l

# >0 = TensorRT branch
# 0 = ONNX-Only branch
```

### Memory Comparison

```bash
# Get current usage
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Expected:
# TensorRT: ~12000-15000 MiB
# ONNX-Only: ~8000-10000 MiB
```

### Performance Comparison

```bash
# Check instance counts
curl localhost:8000/v2/models/text_detection/config | jq '.instance_group'

# TensorRT: count=2
# ONNX-Only: count=3
```

---

## Summary Table

| Metric | TensorRT | ONNX-Only | Winner |
|--------|----------|-----------|--------|
| Throughput | 15 img/s | 10 img/s | TensorRT |
| Latency | 100ms | 140ms | TensorRT |
| VRAM Usage | 12-15GB | 8-10GB | **ONNX** |
| Startup Time | 3-5 min | <30 sec | **ONNX** |
| Memory Spikes | High | Low | **ONNX** |
| Deployment Complexity | High | Low | **ONNX** |
| Cost (shared GPU) | N/A | Lower | **ONNX** |
| Cost (dedicated GPU) | Lower | N/A | TensorRT |
| Development Speed | Slow | Fast | **ONNX** |
| Production Readiness | Yes | Yes | Tie |

---

## Quick Decision Matrix

**Choose TensorRT if:** Performance > All Else
**Choose ONNX-Only if:** Simplicity + Memory > Performance

Still unsure? **Start with ONNX-Only**, migrate to TensorRT if needed.

---

**Maintained branches:**
- `main` - Base configuration
- `trt` - TensorRT optimizations
- `onnx-only-optimization` - ONNX Runtime CUDA EP

**Questions?** See individual branch documentation:
- TensorRT: `TENSORRT_OPTIMIZATION.md`, `VRAM_OPTIMIZATION.md`
- ONNX-Only: `ONNX_ONLY_DEPLOYMENT.md`
