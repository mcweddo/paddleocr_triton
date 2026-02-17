#!/bin/bash

# Clear TensorRT Engine Cache
# Run this script when you change model configs, optimization profiles, or TensorRT settings
# to force TensorRT to rebuild engines with new parameters

CACHE_DIR="/workspace/models/_trt_cache"

echo "=========================================="
echo "TensorRT Cache Cleanup Script"
echo "=========================================="
echo ""

if [ ! -d "$CACHE_DIR" ]; then
    echo "Cache directory not found: $CACHE_DIR"
    echo "Creating directory..."
    mkdir -p "$CACHE_DIR/text_detection"
    mkdir -p "$CACHE_DIR/text_recognition"
    exit 0
fi

echo "Current cache contents:"
du -sh $CACHE_DIR/* 2>/dev/null || echo "  (empty)"
echo ""

read -p "Clear TensorRT cache? This will force engine rebuilds on next startup. (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Clearing cache..."

    rm -rf $CACHE_DIR/text_detection/*
    rm -rf $CACHE_DIR/text_recognition/*

    echo "✓ Cache cleared successfully"
    echo ""
    echo "Next Triton startup will rebuild TensorRT engines."
    echo "This may take 2-5 minutes depending on your GPU."
else
    echo "Cache cleanup cancelled."
fi

echo ""
echo "Done."
