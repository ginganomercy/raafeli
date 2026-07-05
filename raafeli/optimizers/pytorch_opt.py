def optimize_pytorch_model(model, precision: str = "int8"):
    """
    Applies Dynamic Quantization to a PyTorch model.
    Attaches the optimized model to the original model to prevent memory leaks.
    """
    import torch
    
    # Avoid re-optimizing the same model
    if getattr(model, "_raafeli_optimized", False):
        return model
        
    # Security Patch 2: Device Guard (CUDA Check)
    try:
        first_param = next(model.parameters())
        if first_param.device.type == "cuda":
            print("[Raafeli Warning] Model is on GPU. CPU Quantization bypassed.")
            model._raafeli_optimized = True
            model._raafeli_cached_quantized = model
            return model
    except StopIteration:
        pass # Model has no parameters
        
    if precision == "int8":
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            
            # Attach properties to the new model and the original model
            quantized_model._raafeli_optimized = True
            quantized_model._raafeli_cached_quantized = quantized_model
            model._raafeli_cached_quantized = quantized_model
            
            return quantized_model
        except Exception as e:
            print(f"[Raafeli Warning] Quantization failed: {e}. Falling back to original model.")
            model._raafeli_optimized = True
            model._raafeli_cached_quantized = model
            return model
            
    # Default fallback
    model._raafeli_optimized = True
    model._raafeli_cached_quantized = model
    return model
