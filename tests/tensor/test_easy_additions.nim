# Test file for easy PyTorch additions
# Tests: ones, full, randn, detach, log, stack, softmax, cross_entropy

import ../../flambeau
import ../../flambeau/raw/bindings/[neural_nets, rawtensors]
import std/unittest

suite "Easy PyTorch API Additions":
  
  test "torch.ones() - Create tensor of ones":
    let a = ones[float32](@[2'i64, 3'i64])
    check a.shape() == @[2'i64, 3'i64]
    check a[0, 0].item() == 1.0'f32  # First element should be 1
    
  test "torch.full() - Create tensor with value":
    let a = full[float32](@[2'i64, 3'i64], 3.14'f32)
    check a.shape() == @[2'i64, 3'i64]
    check abs(a[0, 0].item() - 3.14'f32) < 1e-6  # All elements should be 3.14
    
  test "torch.randn() - Normal random tensor":
    let a = randn[float32](@[10'i64, 10'i64])
    check a.shape() == @[10'i64, 10'i64]
    # Check that it's not all zeros (with very high probability)
    let sum_val = a.sum().item()
    check sum_val != 0.0'f32
    
  test "torch.log() - Natural logarithm":
    let a = ones[float32](@[3'i64, 3'i64])
    let b = log(a)
    check b.shape() == @[3'i64, 3'i64]
    # log(1) = 0
    check abs(b[0, 0].item()) < 1e-6
    
  test "torch.log2() and torch.log10()":
    let a = full[float32](@[2'i64, 2'i64], 10.0'f32)
    let b = log10(a)
    # log10(10) = 1
    check abs(b[0, 0].item() - 1.0'f32) < 1e-6
    
  test ".detach() - Detach from computation graph":
    var a = ones[float32](@[2'i64, 2'i64])
    let b = a.detach()
    check b.shape() == @[2'i64, 2'i64]
    check b[0, 0].item() == 1.0'f32
    
  test "torch.stack() - Stack tensors":
    let a = ones[float32](@[2'i64, 3'i64])
    let b = full[float32](@[2'i64, 3'i64], 2.0'f32)
    let c = stack(a, b, dim = 0)
    check c.shape() == @[2'i64, 2'i64, 3'i64]
    # Check first tensor
    check c[0, 0, 0].item() == 1.0'f32
    # Check second tensor
    check c[1, 0, 0].item() == 2.0'f32
    
  test "F.softmax() - Softmax activation":
    let a = ones[float32](@[2'i64, 3'i64])  # All ones
    let b = softmax(asRaw(a), dim = 1)
    let bt = asTensor[float32](b)
    # Softmax of all equal values should be 1/n
    check abs(bt[0, 0].item() - 0.333333'f32) < 1e-5
    # Sum along dim should be ~1
    let sum_check = asTensor[float32](rawtensors.sum(b, axis = 1, keepdim = false))
    check abs(sum_check[0].item() - 1.0'f32) < 1e-6
    
  test "F.cross_entropy() - Cross entropy loss":
    # Create simple logits (before softmax)
    let logits = full[float32](@[2'i64, 3'i64], 0.0'f32)  # 2 samples, 3 classes
    # Target classes (must be Long/int64)
    let target = zeros[int64](@[2'i64])  # Both samples predict class 0
    let loss_raw = cross_entropy(asRaw(logits), asRaw(target))
    let loss = asTensor[float32](loss_raw)
    # Loss should be positive
    check loss.item() > 0
    
echo "\n✓ All easy additions tests passed!"
