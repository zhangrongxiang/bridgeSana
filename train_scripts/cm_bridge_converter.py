#!/usr/bin/env python
"""
CM-Bridge Converter: Geometric transformation between TrigFlow and Linear Bridge
Based on the formulation in new-cm-bridge.md

This module implements the geometric transformation that allows sCM models
trained on TrigFlow geometry to be converted to Linear Bridge geometry without loss.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class CMBridgeConverter:
    """
    Converter between TrigFlow (t ∈ [0, π/2]) and Linear Flow (τ ∈ [0,1]).

    TrigFlow:
        x_trig(t) = cos(t)x0 + sin(t)x1
        u_trig(t) = -sin(t)x0 + cos(t)x1

    Linear Flow:
        x_lin(τ) = (1-τ)x0 + τx1
        v_lin = x1 - x0

    Key transformations:
        - Time mapping: τ = T(t) = sin(t) / (sin(t) + cos(t))
        - State projection: h_in = x_trig / (sin(t) + cos(t))
        - Vector field: u_trig = α(t) x_trig + β(t) v_lin
    """

    def __init__(self, sigma_data: float = 1.0):
        """
        Args:
            sigma_data: Latent space scale constant (default: 1.0)
        """
        self.sigma_data = sigma_data

    def time_trig_to_linear(self, t: torch.Tensor) -> torch.Tensor:
        """
        Map TrigFlow time t ∈ [0, π/2] to Linear time τ ∈ [0, 1].

        τ = T(t) = sin(t) / (sin(t) + cos(t))

        Properties:
            - T(0) = 0
            - T(π/4) = 0.5
            - T(π/2) = 1
            - Monotonically increasing
            - Matches SNR: tan²(t) = τ²/(1-τ)²

        Args:
            t: TrigFlow time tensor, shape [...], values in [0, π/2]

        Returns:
            τ: Linear time tensor, shape [...], values in [0, 1]
        """
        sin_t = torch.sin(t)
        cos_t = torch.cos(t)
        tau = sin_t / (sin_t + cos_t + 1e-8)  # Add epsilon for numerical stability
        return tau

    def time_linear_to_trig(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Map Linear time τ ∈ [0, 1] to TrigFlow time t ∈ [0, π/2].

        Inverse of time_trig_to_linear:
        t = arctan(τ / (1-τ))

        Args:
            tau: Linear time tensor, shape [...], values in [0, 1]

        Returns:
            t: TrigFlow time tensor, shape [...], values in [0, π/2]
        """
        # t = arctan(τ / (1-τ))
        # To handle τ=1, we use: t = arctan2(τ, 1-τ)
        t = torch.atan2(tau, 1 - tau + 1e-8)
        return t

    def compute_scale_factor(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute scale factor s(t) = 1 / (sin(t) + cos(t)).

        This maps the circular arc (TrigFlow) to the linear chord (Linear Flow).

        Args:
            t: TrigFlow time tensor, shape [...]

        Returns:
            s: Scale factor, shape [...]
        """
        sin_t = torch.sin(t)
        cos_t = torch.cos(t)
        s = 1.0 / (sin_t + cos_t + 1e-8)
        return s

    def project_state_trig_to_linear(
        self,
        x_trig: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Project TrigFlow state to Linear Flow input space.

        h_in(t) = s(t) · x_trig(t) = x_trig / (sin(t) + cos(t))

        This projection ensures that h_in has the same scale as x_lin(τ).

        Args:
            x_trig: TrigFlow state, shape [B, C, H, W]
            t: TrigFlow time, shape [B] or [B, 1, 1, 1]

        Returns:
            h_in: Projected state for Linear Flow model input, shape [B, C, H, W]
        """
        # Ensure t has correct shape for broadcasting
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)

        s = self.compute_scale_factor(t)
        h_in = s * x_trig
        return h_in

    def compute_vector_field_coefficients(
        self,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute coefficients α(t) and β(t) for vector field transformation.

        u_trig(t) = α(t) x_trig(t) + β(t) v_lin

        where:
            α(t) = (cos(t) - sin(t)) / (sin(t) + cos(t))
            β(t) = 1 / (sin(t) + cos(t))

        Args:
            t: TrigFlow time, shape [...]

        Returns:
            alpha: Coefficient α(t), shape [...]
            beta: Coefficient β(t), shape [...]
        """
        sin_t = torch.sin(t)
        cos_t = torch.cos(t)
        denom = sin_t + cos_t + 1e-8

        alpha = (cos_t - sin_t) / denom
        beta = 1.0 / denom

        return alpha, beta

    def convert_velocity_linear_to_trig(
        self,
        v_linear: torch.Tensor,
        x_trig: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert Linear Flow velocity to TrigFlow velocity.

        Given v_linear (predicted by Bridge model on linear geometry),
        convert to TrigFlow velocity u_trig using:

        u_trig = α(t) x_trig + β(t) v_linear

        where v_linear ≈ v_lin = x1 - x0 (constant velocity field).

        Args:
            v_linear: Linear flow velocity prediction, shape [B, C, H, W]
            x_trig: Current TrigFlow state, shape [B, C, H, W]
            t: TrigFlow time, shape [B] or [B, 1, 1, 1]

        Returns:
            u_trig: TrigFlow velocity, shape [B, C, H, W]
        """
        # Ensure t has correct shape
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)

        alpha, beta = self.compute_vector_field_coefficients(t)

        # Reshape for broadcasting
        alpha = alpha.view(-1, 1, 1, 1)
        beta = beta.view(-1, 1, 1, 1)

        u_trig = alpha * x_trig + beta * v_linear
        return u_trig

    def convert_velocity_trig_to_linear(
        self,
        u_trig: torch.Tensor,
        x_trig: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert TrigFlow velocity to Linear Flow velocity.

        Inverse transformation: given u_trig, solve for v_linear:
        v_linear = (u_trig - α(t) x_trig) / β(t)

        Args:
            u_trig: TrigFlow velocity, shape [B, C, H, W]
            x_trig: Current TrigFlow state, shape [B, C, H, W]
            t: TrigFlow time, shape [B] or [B, 1, 1, 1]

        Returns:
            v_linear: Linear flow velocity, shape [B, C, H, W]
        """
        # Ensure t has correct shape
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)

        alpha, beta = self.compute_vector_field_coefficients(t)

        # Reshape for broadcasting
        alpha = alpha.view(-1, 1, 1, 1)
        beta = beta.view(-1, 1, 1, 1)

        v_linear = (u_trig - alpha * x_trig) / (beta + 1e-8)
        return v_linear

    def compute_reference_velocity_field(
        self,
        x_trig: torch.Tensor,
        t: torch.Tensor,
        bridge_model_prediction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reference velocity field û for TrigFlow using Bridge model prediction.

        This is the core conversion function for using a Bridge model (trained on linear flow)
        to generate velocity predictions for TrigFlow sampling.

        Process:
        1. Project x_trig to linear space: h_in = s(t) · x_trig
        2. Map time: τ = T(t)
        3. Get Bridge prediction: v_φ(h_in, τ, x0) [done externally]
        4. Convert to TrigFlow: û = α(t) x_trig/σ_d + β(t) v_φ/σ_d

        Args:
            x_trig: TrigFlow state, shape [B, C, H, W]
            t: TrigFlow time, shape [B] or [B, 1, 1, 1]
            bridge_model_prediction: Bridge model output v_φ(h_in, τ, x0), shape [B, C, H, W]
                                    This should be the raw prediction without normalization.

        Returns:
            u_hat: Reference velocity field for TrigFlow, shape [B, C, H, W]
        """
        # Ensure t has correct shape
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)

        alpha, beta = self.compute_vector_field_coefficients(t)

        # Reshape for broadcasting
        alpha = alpha.view(-1, 1, 1, 1)
        beta = beta.view(-1, 1, 1, 1)

        # Apply normalization if needed (matching the document formula)
        # û(x_t, t, x0) = α(t) x_t/σ_d + β(t) v_φ/σ_d
        u_hat = alpha * (x_trig / self.sigma_data) + beta * (bridge_model_prediction / self.sigma_data)

        return u_hat

    def verify_transformation(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Verify the geometric transformation is correct.

        This function checks:
        1. Time mapping is monotonic and has correct boundary values
        2. State projection preserves SNR
        3. Vector field transformation is invertible

        Args:
            x0: Source state, shape [B, C, H, W]
            x1: Target state, shape [B, C, H, W]
            t: TrigFlow time to test, shape [B]
            noise: Optional noise for stochastic bridge, shape [B, C, H, W]

        Returns:
            dict with verification results and metrics
        """
        results = {}

        # Ensure t has batch dimension
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # 1. Verify time mapping
        tau = self.time_trig_to_linear(t)
        t_reconstructed = self.time_linear_to_trig(tau)
        time_error = torch.abs(t - t_reconstructed).max().item()
        results['time_mapping_error'] = time_error
        results['tau'] = tau

        # 2. Construct states
        t_expanded = t.view(-1, 1, 1, 1)
        sin_t = torch.sin(t_expanded)
        cos_t = torch.cos(t_expanded)

        # TrigFlow state
        x_trig = cos_t * x0 + sin_t * x1
        if noise is not None:
            x_trig = x_trig + torch.sqrt(2 * sin_t * cos_t) * noise

        # Linear Flow state
        tau_expanded = tau.view(-1, 1, 1, 1)
        x_lin = (1 - tau_expanded) * x0 + tau_expanded * x1
        if noise is not None:
            x_lin = x_lin + torch.sqrt(tau_expanded * (1 - tau_expanded)) * noise

        # 3. Verify state projection
        h_in = self.project_state_trig_to_linear(x_trig, t)
        s = self.compute_scale_factor(t)

        # Check if projected state has similar norm to linear state
        norm_h_in = torch.norm(h_in.reshape(h_in.shape[0], -1), dim=1).mean()
        norm_x_lin = torch.norm(x_lin.reshape(x_lin.shape[0], -1), dim=1).mean()
        results['state_norm_ratio'] = (norm_h_in / (norm_x_lin + 1e-8)).item()

        # 4. Verify vector field transformation
        # True TrigFlow velocity
        u_trig_true = -sin_t * x0 + cos_t * x1

        # True Linear velocity
        v_lin_true = x1 - x0

        # Convert linear to trig
        u_trig_converted = self.convert_velocity_linear_to_trig(v_lin_true, x_trig, t)

        # Check conversion error
        velocity_error = F.mse_loss(u_trig_converted, u_trig_true).item()
        results['velocity_conversion_error'] = velocity_error

        # 5. Verify invertibility
        v_lin_reconstructed = self.convert_velocity_trig_to_linear(u_trig_converted, x_trig, t)
        invertibility_error = F.mse_loss(v_lin_reconstructed, v_lin_true).item()
        results['invertibility_error'] = invertibility_error

        # 6. Cosine similarity
        u_trig_true_flat = u_trig_true.reshape(u_trig_true.shape[0], -1)
        u_trig_converted_flat = u_trig_converted.reshape(u_trig_converted.shape[0], -1)
        cos_sim = F.cosine_similarity(u_trig_true_flat, u_trig_converted_flat, dim=1).mean().item()
        results['velocity_cosine_similarity'] = cos_sim

        return results


def create_converter(sigma_data: float = 1.0) -> CMBridgeConverter:
    """
    Factory function to create a CMBridgeConverter instance.

    Args:
        sigma_data: Latent space scale constant

    Returns:
        CMBridgeConverter instance
    """
    return CMBridgeConverter(sigma_data=sigma_data)
