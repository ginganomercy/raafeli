import torch
import torch.nn as nn
import time
from raafeli import optimize_cpu

class HeavyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A massive multi-layer perceptron to simulate heavy load
        layers = []
        in_features = 2048
        for _ in range(8):
            layers.append(nn.Linear(in_features, 4096))
            layers.append(nn.ReLU())
            in_features = 4096
        layers.append(nn.Linear(in_features, 1000))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def run_simulation():
    print("Initializing Heavy Model (FP32)...")
    model = HeavyModel()
    
    # Batch of 64 requests
    dummy_input = torch.randn(64, 2048)
    
    # 1. Standard Prediction
    print("\n--- STANDARD PREDICTION (FP32) ---")
    start = time.perf_counter()
    out_standard = model(dummy_input)
    end = time.perf_counter()
    time_fp32 = end - start
    print(f"Time taken: {time_fp32:.4f} seconds")
    
    # 2. Optimized Prediction
    @optimize_cpu(model_arg="model")
    def optimized_predict(model, data):
        return model(data)
        
    print("\n--- RAAFELI PREDICTION (INT8) ---")
    # First run will quantize
    print("Running initial quantization (cold start)...")
    out_opt = optimized_predict(model, dummy_input)
    
    start = time.perf_counter()
    out_opt2 = optimized_predict(model, dummy_input)
    end = time.perf_counter()
    time_int8 = end - start
    print(f"Time taken: {time_int8:.4f} seconds")
    
    print("\n--- RESULTS ---")
    speedup = time_fp32 / time_int8
    print(f"Raafeli is {speedup:.2f}x faster on CPU!")
    
if __name__ == "__main__":
    run_simulation()
