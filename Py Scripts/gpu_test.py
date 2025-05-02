# Save as gpu_test.py
def check_gpu():
    """Test all GPU libraries"""
    print("=== GPU Library Check ===\n")
    
    # Test CUDA and driver
    try:
        import pycuda.driver as cuda
        cuda.init()
        print(f"PyCUDA: ✓ ({cuda.get_version()})")
        device = cuda.Device(0)
        print(f"  -> Device: {device.name()}")
        print(f"  -> Memory: {device.total_memory() / 1e9:.2f} GB")
    except Exception as e:
        print(f"PyCUDA: ✗ ({str(e)})")
    
    # Test CuPy
    try:
        import cupy as cp
        x = cp.array([1, 2, 3])
        result = cp.sum(x)
        print(f"CuPy: ✓ (version {cp.__version__})")
        print(f"  -> Test calculation: {result}")
    except Exception as e:
        print(f"CuPy: ✗ ({str(e)})")
    
    # Test PyTorch with CUDA
    try:
        import torch
        print(f"PyTorch: ✓ (version {torch.__version__})")
        print(f"  -> CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  -> Device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"PyTorch: ✗ ({str(e)})")
    
    # Test RAPIDS (cuDF & cuGraph)
    try:
        import cudf
        test_df = cudf.DataFrame({'a': [1, 2, 3]})
        result = test_df.a.sum()
        print(f"cuDF: ✓ (version {cudf.__version__})")
        print(f"  -> Test calculation: {result}")
        
        import cugraph
        edges = cudf.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
        G = cugraph.DiGraph()
        G.from_cudf_edgelist(edges, 'src', 'dst')
        print(f"cuGraph: ✓ (version {cugraph.__version__})")
        print(f"  -> Test graph: {G.number_of_vertices()} vertices, {G.number_of_edges()} edges")
    except Exception as e:
        print(f"RAPIDS: ✗ ({str(e)})")

if __name__ == "__main__":
    check_gpu()