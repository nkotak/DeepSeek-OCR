"""MLP Projector for Vision-Language Models (MLX Implementation)

This module implements various MLP projector types for connecting vision encoders
to language models. It uses MLX native operations including unfold for downsampling.

Projector Types:
    - identity: Pass-through (no transformation)
    - linear: Simple linear projection
    - mlp_gelu: Multi-layer MLP with GELU activation
    - downsample_mlp_gelu: Downsampling with unfold + MLP
    - normlayer_downsample_mlp_gelu: LayerNorm + downsampling + MLP
    - low_high_hybrid_split_mlp_gelu: Separate projections for high/low features
    - hybrid_split_feature_mlp_gelu: Split features by channel dimension
    - low_high_split_mlp_gelu: Separate processing paths for high/low

References:
    - Implementation based on: DeepSeek-OCR-vllm/deepencoder/build_linear.py
    - Uses MLX native operations (unfold_mlx for downsampling)
"""
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List, Union, Dict, Any
import math

from .utils_mlx import unfold_mlx


class MlpProjector(nn.Module):
    """Multi-layer perceptron projector with various architecture types"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary with keys:
                - projector_type: Type of projector architecture
                - input_dim: Input dimension(s)
                - n_embed: Output embedding dimension
                - depth: Number of MLP layers (optional, default=1)
                - mlp_ratio: MLP hidden dimension ratio (optional, default=1)
                - downsample_ratio: Downsampling ratio for unfold (optional, default=2)
                - token_pooling: Whether to use token pooling (optional, default=False)
                - conv_fusion_high_low_features: Fusion layer (optional, default=False)
                - channel_div: Channel division ratio (optional, default=0.5)
        """
        super().__init__()
        self.config = config
        self.projector_type = config["projector_type"]

        # Build projector layers based on type
        if self.projector_type == "identity":
            self.layers = nn.Identity()

        elif self.projector_type == "linear":
            self.layers = nn.Linear(config["input_dim"], config["n_embed"])

        elif self.projector_type == "mlp_gelu":
            mlp_depth = config.get("depth", 1)
            modules = [nn.Linear(config["input_dim"], config["n_embed"])]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config["n_embed"], config["n_embed"]))
            self.layers = modules

        elif self.projector_type == "normlayer_downsample_mlp_gelu":
            mlp_depth = config.get("depth", 1)
            mlp_ratio = config.get("mlp_ratio", 1)
            downsample_ratio = config.get("downsample_ratio", 2)

            # Input dimension after downsampling: input_dim * downsample_ratio^2
            downsampled_dim = config["input_dim"] * downsample_ratio * downsample_ratio

            modules = [
                nn.LayerNorm(downsampled_dim),
                nn.Linear(downsampled_dim, config["n_embed"] * mlp_ratio)
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config["n_embed"] * mlp_ratio, config["n_embed"] * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(config["n_embed"] * mlp_ratio, config["n_embed"]))
            self.layers = modules

        elif self.projector_type == "downsample_mlp_gelu":
            mlp_depth = config.get("depth", 1)
            mlp_ratio = config.get("mlp_ratio", 1)
            downsample_ratio = config.get("downsample_ratio", 2)

            # Input dimension after downsampling: input_dim * downsample_ratio^2
            downsampled_dim = config["input_dim"] * downsample_ratio * downsample_ratio

            modules = [nn.Linear(downsampled_dim, config["n_embed"] * mlp_ratio)]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config["n_embed"] * mlp_ratio, config["n_embed"] * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(config["n_embed"] * mlp_ratio, config["n_embed"]))
            self.layers = modules

        elif self.projector_type == "low_high_hybrid_split_mlp_gelu":
            mlp_depth = config.get("depth", 1)
            self.high_up_proj = nn.Linear(config["input_dim"], config["n_embed"] // 2)
            self.low_up_proj = nn.Linear(config["input_dim"], config["n_embed"] // 2)

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config["n_embed"], config["n_embed"]))
            self.layers = modules

        elif self.projector_type == "hybrid_split_feature_mlp_gelu":
            mlp_depth = config.get("depth", 1)
            channel_div = config.get("channel_div", 0.5)
            input_dim = config["input_dim"]

            # Expect input_dim to be a list [high_dim, low_dim]
            if isinstance(input_dim, list):
                high_dim, low_dim = input_dim[0], input_dim[1]
            else:
                # Split single input dimension
                high_dim = low_dim = input_dim

            self.high_up_proj = nn.Linear(high_dim, int(config["n_embed"] * channel_div))
            self.low_up_proj = nn.Linear(low_dim, config["n_embed"] - int(config["n_embed"] * channel_div))

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config["n_embed"], config["n_embed"]))
            self.layers = modules

        elif self.projector_type == "low_high_split_mlp_gelu":
            mlp_depth = config.get("depth", 1)
            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config["n_embed"] // 2, config["n_embed"] // 2))

            self.high_layers = modules
            # Deep copy for separate weights
            self.low_layers = [
                nn.GELU() if isinstance(m, nn.GELU) else
                nn.Linear(config["n_embed"] // 2, config["n_embed"] // 2)
                for m in modules
            ]
            self.layers = None  # Not used in this mode

        else:
            raise ValueError(f"Unknown projector type: {self.projector_type}")

        # Optional token pooling layer
        if config.get("token_pooling", False):
            self.token_pooling_layer = nn.Linear(config["input_dim"] * 4, config["input_dim"])
        else:
            self.token_pooling_layer = None

        # Optional fusion layer
        if config.get("conv_fusion_high_low_features", False):
            self.fusion_layer = nn.Linear(config["input_dim"], config["input_dim"])
        else:
            self.fusion_layer = None

    def __call__(self, x: Union[mx.array, List[mx.array]]) -> mx.array:
        """
        Forward pass through projector

        Args:
            x: Input tensor(s)
                - For most types: [B, L, C]
                - For split types: List of [high_features, low_features]

        Returns:
            Projected features [B, L', C_out]
        """
        # Token pooling (2x2 patch merging using unfold)
        if self.token_pooling_layer is not None:
            batch_size, wxh, channels = x.shape
            w = h = int(wxh ** 0.5)

            # Reshape to spatial: [B, L, C] -> [B, H, W, C] -> [B, C, H, W]
            x = x.reshape([batch_size, w, h, channels])
            x = x.transpose([0, 3, 1, 2])

            # Unfold 2x2 patches: [B, C, H, W] -> [B, C*4, H/2*W/2]
            patches = unfold_mlx(x, kernel_size=2, stride=2, padding=0)

            # Reshape: [B, C*4, L/4] -> [B, L/4, C*4]
            patches = patches.transpose([0, 2, 1])

            # Apply pooling layer
            x = self.token_pooling_layer(patches)

        # Fusion layer (for high-low feature fusion)
        if self.fusion_layer is not None:
            # Assume x is a list [high, low] or has multiple elements in first dim
            x = self.fusion_layer(x[:, 0]) + x[:, 1]

        # Handle split projector types
        if self.projector_type == 'low_high_hybrid_split_mlp_gelu':
            high_x, low_x = x[0], x[1]
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = mx.concatenate([high_x, low_x], axis=-1)

        elif self.projector_type == 'hybrid_split_feature_mlx_gelu':
            # Split by channel dimension
            if isinstance(self.config["input_dim"], list):
                high_dim = self.config["input_dim"][0]
                high_x = x[..., :high_dim]
                low_x = x[..., high_dim:]
            else:
                # Split evenly
                mid = x.shape[-1] // 2
                high_x = x[..., :mid]
                low_x = x[..., mid:]

            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = mx.concatenate([high_x, low_x], axis=-1)

        elif self.projector_type == 'low_high_split_mlp_gelu':
            high_x, low_x = x[0], x[1]

            # Process through separate layers
            for layer in self.high_layers:
                if isinstance(layer, nn.GELU):
                    high_x = layer(high_x)
                else:
                    high_x = layer(high_x)

            for layer in self.low_layers:
                if isinstance(layer, nn.GELU):
                    low_x = layer(low_x)
                else:
                    low_x = layer(low_x)

            x = mx.concatenate([high_x, low_x], axis=-1)
            return x

        # Handle downsampling for downsample_mlp_gelu types
        if self.projector_type in ['downsample_mlp_gelu', 'normlayer_downsample_mlp_gelu']:
            bs, hw, input_dim = x.shape
            h = w = int(hw ** 0.5)

            downsample_ratio = self.config.get("downsample_ratio", 2)

            # Compute padding
            if h % downsample_ratio:
                pad = downsample_ratio - h % downsample_ratio
            else:
                pad = 0

            # Reshape to spatial: [B, H*W, C] -> [B, H, W, C]
            x = x.reshape([bs, h, w, input_dim])

            # Pad if necessary
            if pad > 0:
                # MLX padding format: [(before_dim0, after_dim0), (before_dim1, after_dim1), ...]
                x = mx.pad(x, [(0, 0), (0, pad), (0, pad), (0, 0)], constant_values=0)

            # Convert to [B, C, H, W] for unfold
            x = x.transpose([0, 3, 1, 2])

            # Unfold: 4-to-1 concat pattern (e.g., 2x2 patches -> 4*C channels)
            # [B, C, H, W] -> [B, C*downsample_ratio^2, H/downsample_ratio * W/downsample_ratio]
            x = unfold_mlx(x, kernel_size=downsample_ratio, stride=downsample_ratio, padding=0)

            # Transpose: [B, C*4, L/4] -> [B, L/4, C*4]
            x = x.transpose([0, 2, 1])

        # Apply main layers
        if isinstance(self.layers, list):
            # Sequential application
            for layer in self.layers:
                x = layer(x)
        else:
            # Single layer or Identity
            x = self.layers(x)

        return x


def build_linear_projector(input_dim: int, n_embed: int) -> MlpProjector:
    """
    Build a simple linear projector (most common configuration)

    Args:
        input_dim: Input dimension (typically 2048 for SAM+CLIP concatenation)
        n_embed: Output embedding dimension (language model dimension)

    Returns:
        MlpProjector instance
    """
    config = {
        "projector_type": "linear",
        "input_dim": input_dim,
        "n_embed": n_embed,
    }
    return MlpProjector(config)


def build_downsample_projector(
    input_dim: int,
    n_embed: int,
    downsample_ratio: int = 2,
    depth: int = 2,
    mlp_ratio: int = 1,
    use_norm: bool = False
) -> MlpProjector:
    """
    Build a downsampling projector with unfold operation

    Args:
        input_dim: Input dimension per token
        n_embed: Output embedding dimension
        downsample_ratio: Spatial downsampling ratio (default=2 for 4-to-1)
        depth: Number of MLP layers
        mlp_ratio: MLP hidden dimension multiplier
        use_norm: Whether to use LayerNorm before MLP

    Returns:
        MlpProjector instance
    """
    projector_type = "normlayer_downsample_mlp_gelu" if use_norm else "downsample_mlp_gelu"

    config = {
        "projector_type": projector_type,
        "input_dim": input_dim,
        "n_embed": n_embed,
        "downsample_ratio": downsample_ratio,
        "depth": depth,
        "mlp_ratio": mlp_ratio,
    }
    return MlpProjector(config)
