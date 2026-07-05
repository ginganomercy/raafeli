import functools
import inspect
from typing import Any

from .optimizers.pytorch_opt import optimize_pytorch_model

def optimize_cpu(model_arg: str = "model", precision: str = "int8"):
    """
    Decorator to automatically optimize a PyTorch model for CPU inference
    using Dynamic Quantization.
    """
    def decorator(func):
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Security Patch 1: Silent Error Guard
            if model_arg not in bound_args.arguments:
                raise ValueError(f"[Raafeli Error] Argument '{model_arg}' not found in function '{func.__name__}'. Please check your decorator arguments.")
                
            model = bound_args.arguments[model_arg]
            
            # Check if it's a PyTorch model
            if hasattr(model, "parameters") and hasattr(model, "forward"):
                # Security Patch 3: OOP Caching (No Global Dicts)
                if hasattr(model, "_raafeli_cached_quantized"):
                    optimized_model = model._raafeli_cached_quantized
                else:
                    optimized_model = optimize_pytorch_model(model, precision)
                    
                bound_args.arguments[model_arg] = optimized_model
                    
            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator
