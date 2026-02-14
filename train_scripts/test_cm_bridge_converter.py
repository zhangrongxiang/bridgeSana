#!/usr/bin/env python
"""
Unit tests for CM-Bridge Converter
Tests individual transformation functions without requiring trained models.
"""

import torch
import torch.nn.functional as F
import numpy as np

from cm_bridge_converter import CMBridgeConverter


def test_time_mapping():
    """Test time mapping between TrigFlow and Linear"""
    print("\n" + "="*60)
    print("Test 1: Time Mapping T(t): TrigFlow → Linear")
    print("="*60)

    converter = CMBridgeConverter()

    # Test boundary values
    print("\nBoundary value tests:")
    test_cases = [
        (0.0, 0.0, "T(0) = 0"),
        (np.pi/4, 0.5, "T(π/4) = 0.5"),
        (np.pi/2, 1.0, "T(π/2) = 1"),
    ]

    all_passed = True
    for t_val, expected_tau, desc in test_cases:
        t = torch.tensor(t_val)
        tau = converter.time_trig_to_linear(t)
        error = abs(tau.item() - expected_tau)
        passed = error < 1e-6

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {desc}")
        print(f"         t={t_val:.6f} → tau={tau.item():.6f} (expected {expected_tau:.6f}, error={error:.2e})")

        all_passed = all_passed and passed

    # Test monotonicity
    print("\nMonotonicity test:")
    t_values = torch.linspace(0, np.pi/2, 100)
    tau_values = converter.time_trig_to_linear(t_values)
    is_monotonic = torch.all(tau_values[1:] >= tau_values[:-1])
    print(f"  {'✅ PASS' if is_monotonic else '❌ FAIL'}: tau is monotonically increasing")

    # Test invertibility
    print("\nInvertibility test:")
    t_test = torch.linspace(0.01, np.pi/2 - 0.01, 20)
    tau = converter.time_trig_to_linear(t_test)
    t_reconstructed = converter.time_linear_to_trig(tau)
    max_error = torch.max(torch.abs(t_test - t_reconstructed)).item()
    passed = max_error < 1e-5

    print(f"  {'✅ PASS' if passed else '❌ FAIL'}: Time mapping is invertible")
    print(f"         Max reconstruction error: {max_error:.2e}")

    all_passed = all_passed and is_monotonic and passed

    return all_passed


def test_snr_matching():
    """Test that SNR is preserved under time mapping"""
    print("\n" + "="*60)
    print("Test 2: SNR Matching")
    print("="*60)

    converter = CMBridgeConverter()

    # Create random source and target
    torch.manual_seed(42)
    x0 = torch.randn(2, 4, 8, 8)
    x1 = torch.randn(2, 4, 8, 8)

    print("\nComparing SNR at corresponding times:")

    test_times = torch.tensor([0.3, 0.7, 1.0, 1.3])
    all_passed = True

    for t in test_times:
        # Map time
        tau = converter.time_trig_to_linear(t)

        # Compute SNR in TrigFlow
        sin_t = torch.sin(t)
        cos_t = torch.cos(t)
        snr_trig = (sin_t**2) / (cos_t**2)  # Approximate SNR

        # Compute SNR in Linear Flow
        snr_lin = (tau**2) / ((1 - tau)**2)  # Approximate SNR

        # They should be equal: tan²(t) = τ²/(1-τ)²
        error = abs(snr_trig.item() - snr_lin.item())
        passed = error < 1e-5

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: t={t:.4f}, tau={tau:.4f}")
        print(f"         SNR_trig={snr_trig:.6f}, SNR_lin={snr_lin:.6f}, error={error:.2e}")

        all_passed = all_passed and passed

    return all_passed


def test_state_projection():
    """Test state projection from TrigFlow to Linear"""
    print("\n" + "="*60)
    print("Test 3: State Projection")
    print("="*60)

    converter = CMBridgeConverter()

    torch.manual_seed(42)
    x0 = torch.randn(2, 4, 8, 8)
    x1 = torch.randn(2, 4, 8, 8)

    print("\nProjecting TrigFlow states to Linear geometry:")

    test_times = torch.tensor([0.5, 1.0])
    all_passed = True

    for t in test_times:
        t_batch = t.unsqueeze(0).expand(2)
        tau = converter.time_trig_to_linear(t_batch)

        # Construct states
        sin_t = torch.sin(t)
        cos_t = torch.cos(t)
        x_trig = cos_t * x0 + sin_t * x1

        x_lin = (1 - tau.view(-1, 1, 1, 1)) * x0 + tau.view(-1, 1, 1, 1) * x1

        # Project
        h_in = converter.project_state_trig_to_linear(x_trig, t_batch)
        s = converter.compute_scale_factor(t_batch)

        # Check scale factor
        expected_s = 1.0 / (sin_t + cos_t)
        scale_error = abs(s[0].item() - expected_s.item())

        # Compare norms
        norm_h_in = torch.norm(h_in.reshape(h_in.shape[0], -1), dim=1).mean()
        norm_x_lin = torch.norm(x_lin.reshape(x_lin.shape[0], -1), dim=1).mean()
        norm_ratio = (norm_h_in / norm_x_lin).item()

        # The norms should be similar (within 20%)
        passed = abs(norm_ratio - 1.0) < 0.2

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: t={t:.4f}")
        print(f"         Scale factor s={s[0]:.6f} (expected {expected_s:.6f}, error={scale_error:.2e})")
        print(f"         Norm ratio ||h_in|| / ||x_lin|| = {norm_ratio:.4f}")

        all_passed = all_passed and passed

    return all_passed


def test_vector_field_transformation():
    """Test vector field transformation"""
    print("\n" + "="*60)
    print("Test 4: Vector Field Transformation")
    print("="*60)

    converter = CMBridgeConverter()

    torch.manual_seed(42)
    x0 = torch.randn(2, 4, 8, 8)
    x1 = torch.randn(2, 4, 8, 8)

    print("\nTesting u_trig = α(t) x_trig + β(t) v_lin:")

    test_times = torch.tensor([0.3, 0.7, 1.0])
    all_passed = True

    for t in test_times:
        t_batch = t.unsqueeze(0).expand(2)

        # Construct states
        sin_t = torch.sin(t)
        cos_t = torch.cos(t)
        x_trig = cos_t * x0 + sin_t * x1

        # True velocities
        u_trig_true = -sin_t * x0 + cos_t * x1
        v_lin_true = x1 - x0

        # Test conversion: Linear → TrigFlow
        u_trig_converted = converter.convert_velocity_linear_to_trig(v_lin_true, x_trig, t_batch)

        # Compute error
        error = F.mse_loss(u_trig_converted, u_trig_true).item()
        cos_sim = F.cosine_similarity(
            u_trig_converted.reshape(2, -1),
            u_trig_true.reshape(2, -1),
            dim=1
        ).mean().item()

        passed = error < 1e-10 and cos_sim > 0.9999

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: t={t:.4f}")
        print(f"         MSE error={error:.2e}, cosine similarity={cos_sim:.8f}")

        # Test invertibility: TrigFlow → Linear → TrigFlow
        v_lin_reconstructed = converter.convert_velocity_trig_to_linear(u_trig_converted, x_trig, t_batch)
        inv_error = F.mse_loss(v_lin_reconstructed, v_lin_true).item()

        inv_passed = inv_error < 1e-10
        inv_status = "✅ PASS" if inv_passed else "❌ FAIL"
        print(f"  {inv_status}: Invertibility test")
        print(f"         Reconstruction error={inv_error:.2e}")

        all_passed = all_passed and passed and inv_passed

    return all_passed


def test_full_transformation():
    """Test full transformation pipeline"""
    print("\n" + "="*60)
    print("Test 5: Full Transformation Pipeline")
    print("="*60)

    converter = CMBridgeConverter(sigma_data=1.0)

    torch.manual_seed(42)
    x0 = torch.randn(2, 4, 8, 8)
    x1 = torch.randn(2, 4, 8, 8)

    print("\nRunning verify_transformation on multiple timesteps:")

    test_times = torch.tensor([0.2, 0.5, 1.0, 1.3])
    all_passed = True

    for t in test_times:
        results = converter.verify_transformation(x0, x1, t)

        # Check all metrics
        passed = (
            results['time_mapping_error'] < 1e-6 and
            results['velocity_conversion_error'] < 1e-10 and
            results['invertibility_error'] < 1e-10 and
            results['velocity_cosine_similarity'] > 0.9999
        )

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n  {status}: t={t:.4f}")
        print(f"         Time mapping error: {results['time_mapping_error']:.2e}")
        print(f"         Velocity conversion error: {results['velocity_conversion_error']:.2e}")
        print(f"         Invertibility error: {results['invertibility_error']:.2e}")
        print(f"         Cosine similarity: {results['velocity_cosine_similarity']:.8f}")

        all_passed = all_passed and passed

    return all_passed


def main():
    print("\n" + "="*80)
    print("CM-Bridge Converter Unit Tests")
    print("="*80)

    tests = [
        ("Time Mapping", test_time_mapping),
        ("SNR Matching", test_snr_matching),
        ("State Projection", test_state_projection),
        ("Vector Field Transformation", test_vector_field_transformation),
        ("Full Transformation Pipeline", test_full_transformation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n❌ Test failed with exception: {e}")

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    total = len(results)
    passed = sum(1 for _, p, _ in results if p)

    for test_name, test_passed, error in results:
        status = "✅ PASS" if test_passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if error:
            print(f"       Error: {error}")

    print("\n" + "="*80)
    print(f"Results: {passed}/{total} tests passed")
    print("="*80)

    if passed == total:
        print("\n🎉 All tests passed! The CM-Bridge converter is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit(main())
