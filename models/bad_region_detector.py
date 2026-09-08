"""
CNN/Transformer model for detecting bad regions in cryoEM micrographs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional
import numpy as np


class BadRegionDetector(nn.Module):
    """
    CNN-based model for detecting bad regions in micrographs.
    Uses a U-Net-like architecture for pixel-level segmentation.
    
    Can optionally use power spectrum (Fourier space) as additional input channel,
    using complementary real-space and Fourier-space information.
    """
    
    def __init__(self, input_channels: int = 1, num_classes: int = 2, use_power_spectrum: bool = False):
        """
        Initialize the bad region detector.
        
        Args:
            input_channels: Number of input channels (1 for grayscale, 2 for image+power_spectrum, 4 for enhanced Fourier)
            num_classes: Number of output classes (2 for binary: good/bad, 4+ for multi-class)
            use_power_spectrum: If True, expects 2-channel input (image + power spectrum)
        """
        super(BadRegionDetector, self).__init__()
        self.use_power_spectrum = use_power_spectrum
        
        # Encoder (downsampling path)
        self.enc1 = self._conv_block(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = self._conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = self._conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = self._conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder (upsampling path)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Final classification layer (multi-class)
        self.final = nn.Conv2d(64, num_classes, 1)
        
        # Binary badness head (predicts bad vs good as single probability)
        self.binary_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),  # Single channel: probability of badness
        )
        
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolutional block with batch norm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.up4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Final outputs
        multiclass_logits = self.final(x)
        binary_logits = self.binary_head(x)
        
        return multiclass_logits, binary_logits


class BottleneckSelfAttention(nn.Module):
    """
    Self-attention block for use at CNN bottleneck (coarsest resolution).
    
    Provides global context without full ViT overhead. At 1/16 resolution,
    a 128x128 input becomes 8x8 = 64 tokens, which is very efficient.
    """
    
    def __init__(self, embed_dim: int = 1024, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Learnable positional encoding (max 32x32 = 1024 positions)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1024, embed_dim) * 0.02)
        
        # Layer norm before output
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature map at bottleneck resolution
        
        Returns:
            [B, C, H, W] feature map with global context
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dims to sequence: [B, C, H, W] -> [B, H*W, C]
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Add positional encoding
        num_tokens = H * W
        pos = self.pos_encoding[:, :num_tokens, :]  # [1, H*W, C]
        x_flat = x_flat + pos
        
        # Self-attention
        x_attn = self.transformer(x_flat)  # [B, H*W, C]
        x_attn = self.norm(x_attn)
        
        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        x_out = x_attn.transpose(1, 2).view(B, C, H, W)
        
        # Residual connection
        return x + x_out


class DualLocalGlobalAttention(nn.Module):
    """
    Dual Local + Global Attention for bottleneck.
    
    Addresses the edge FP problem by combining:
    1. LOCAL attention (windowed 8x8): Learns texture discrimination
       - Cannot be contaminated by edge artifacts (windows don't cross boundaries)
       - Immune to global biases
       
    2. GLOBAL attention (full grid with edge masking): Learns structure
       - Outer 15% of positions masked (can't attend to/from edges)
       - Provides large-scale context without edge bias
    
    Architecture:
        Input: [B, 1024, H, W]
        ├── Local branch:  Project to 512-dim → Windowed attention → [B, 512, H, W]
        ├── Global branch: Project to 512-dim → Masked attention → [B, 512, H, W]
        └── Concat → [B, 1024, H, W] → Output
    
    This dual approach:
    - Eliminates edge swaths (edge masking in global + local windows don't see edges)
    - Reduces particle FPs (local texture discrimination)
    - Maintains large-scale structure understanding (global attention)
    """
    
    def __init__(
        self, 
        embed_dim: int = 1024, 
        local_heads: int = 4,
        global_heads: int = 4,
        num_layers: int = 2, 
        dropout: float = 0.1,
        window_size: int = 8,
        edge_mask_fraction: float = 0.15,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.branch_dim = embed_dim // 2  # Each branch gets half
        self.window_size = window_size
        self.edge_mask_fraction = edge_mask_fraction
        
        # Project input to branch dimensions
        self.local_proj_in = nn.Linear(embed_dim, self.branch_dim)
        self.global_proj_in = nn.Linear(embed_dim, self.branch_dim)
        
        # Local attention (windowed)
        local_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.branch_dim,
            nhead=local_heads,
            dim_feedforward=self.branch_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.local_transformer = nn.TransformerEncoder(local_encoder_layer, num_layers=num_layers)
        self.local_pos = nn.Parameter(torch.randn(1, window_size * window_size, self.branch_dim) * 0.02)
        self.local_norm = nn.LayerNorm(self.branch_dim)
        
        # Global attention (masked)
        global_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.branch_dim,
            nhead=global_heads,
            dim_feedforward=self.branch_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.global_transformer = nn.TransformerEncoder(global_encoder_layer, num_layers=num_layers)
        self.global_pos = nn.Parameter(torch.randn(1, 1024, self.branch_dim) * 0.02)  # Max 32x32
        self.global_norm = nn.LayerNorm(self.branch_dim)
        
        # Project concatenated output back to embed_dim
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        
        # Cache for edge mask
        self._edge_mask_cache = {}
    
    def _get_edge_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Create edge mask: True for positions to MASK (edges), False for positions to KEEP.
        
        Masks outer `edge_mask_fraction` of positions on each side.
        For 16x16 with 15%: masks outer ~2.4 pixels → round to 2 pixels.
        """
        cache_key = (H, W, device)
        if cache_key in self._edge_mask_cache:
            return self._edge_mask_cache[cache_key]
        
        # Calculate edge width (at least 1 pixel)
        edge_h = max(1, int(H * self.edge_mask_fraction))
        edge_w = max(1, int(W * self.edge_mask_fraction))
        
        # Create mask: True = masked (edge), False = keep (interior)
        mask = torch.zeros(H, W, dtype=torch.bool, device=device)
        mask[:edge_h, :] = True   # Top edge
        mask[-edge_h:, :] = True  # Bottom edge
        mask[:, :edge_w] = True   # Left edge
        mask[:, -edge_w:] = True  # Right edge
        
        # Flatten to [H*W]
        mask = mask.flatten()
        
        self._edge_mask_cache[cache_key] = mask
        return mask
    
    def _windowed_attention(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Apply attention within non-overlapping windows.
        
        Args:
            x: [B, H*W, branch_dim] flattened features
            H, W: Spatial dimensions
            
        Returns:
            [B, H*W, branch_dim] features after windowed attention
        """
        B = x.shape[0]
        ws = self.window_size
        
        # Reshape to spatial: [B, H, W, D]
        x = x.view(B, H, W, self.branch_dim)
        
        # Pad if not divisible by window size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # Pad spatial dims
        
        Hp, Wp = x.shape[1], x.shape[2]
        
        # Partition into windows: [B, num_h, ws, num_w, ws, D] -> [B*num_windows, ws*ws, D]
        num_h, num_w = Hp // ws, Wp // ws
        x = x.view(B, num_h, ws, num_w, ws, self.branch_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, num_h, num_w, ws, ws, D]
        x = x.view(B * num_h * num_w, ws * ws, self.branch_dim)  # [B*num_windows, ws*ws, D]
        
        # Add positional encoding within windows
        x = x + self.local_pos
        
        # Apply attention within each window
        x = self.local_transformer(x)
        x = self.local_norm(x)
        
        # Reverse partition: [B*num_windows, ws*ws, D] -> [B, Hp, Wp, D]
        x = x.view(B, num_h, num_w, ws, ws, self.branch_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, num_h, ws, num_w, ws, D]
        x = x.view(B, Hp, Wp, self.branch_dim)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
        
        # Flatten back: [B, H*W, D]
        x = x.view(B, H * W, self.branch_dim)
        
        return x
    
    def _masked_global_attention(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Apply global attention with edge masking.
        
        Edge positions are masked out - they cannot attend to or be attended from.
        This prevents edge artifacts from contaminating the global context.
        
        Args:
            x: [B, H*W, branch_dim] flattened features
            H, W: Spatial dimensions
            
        Returns:
            [B, H*W, branch_dim] features after masked global attention
        """
        B, N, D = x.shape
        device = x.device
        
        # Get edge mask: [H*W] boolean, True = mask out
        edge_mask = self._get_edge_mask(H, W, device)  # [H*W]
        
        # Add positional encoding
        pos = self.global_pos[:, :N, :]
        x = x + pos
        
        # For TransformerEncoder, we need to use src_key_padding_mask
        # Shape should be [B, N] where True means "ignore this position"
        # But edge_mask is same for all batches, so expand it
        mask_expanded = edge_mask.unsqueeze(0).expand(B, -1)  # [B, N]
        
        # Apply global attention with edge masking
        # Note: src_key_padding_mask=True means "do not attend to this position"
        x = self.global_transformer(x, src_key_padding_mask=mask_expanded)
        x = self.global_norm(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dual local + global attention forward pass.
        
        Args:
            x: [B, C, H, W] feature map at bottleneck resolution
            
        Returns:
            [B, C, H, W] feature map with dual-scale context
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dims: [B, C, H, W] -> [B, H*W, C]
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Project to branch dimensions
        x_local = self.local_proj_in(x_flat)   # [B, H*W, branch_dim]
        x_global = self.global_proj_in(x_flat)  # [B, H*W, branch_dim]
        
        # Branch A: Windowed local attention (immune to edges)
        x_local = self._windowed_attention(x_local, H, W)
        
        # Branch B: Masked global attention (edges masked out)
        x_global = self._masked_global_attention(x_global, H, W)
        
        # Concatenate branches: [B, H*W, branch_dim] + [B, H*W, branch_dim] -> [B, H*W, embed_dim]
        x_concat = torch.cat([x_local, x_global], dim=-1)
        
        # Project back and normalize
        x_out = self.proj_out(x_concat)
        x_out = self.output_norm(x_out)
        
        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        x_out = x_out.transpose(1, 2).view(B, C, H, W)
        
        # Residual connection
        return x + x_out


class BadRegionDetectorWithAttention(nn.Module):
    """
    CNN-based model with self-attention at bottleneck for global context.
    
    Same U-Net architecture as BadRegionDetector, but adds 1-2 self-attention
    layers at the bottleneck (1/16 resolution) to capture long-range dependencies
    like carbon edges spanning the image.
    
    This is a hybrid approach: CNN inductive bias everywhere, attention only
    where it matters for global coherence.
    """
    
    def __init__(
        self, 
        input_channels: int = 1, 
        num_classes: int = 2, 
        use_power_spectrum: bool = False,
        attention_heads: int = 8,
        attention_layers: int = 2,
        attention_dropout: float = 0.1,
        norm_type: str = "batch",  # "batch", "group", or "instance"
        decoder_dropout: float = 0.0,  # NEW: Dropout in decoder to prevent overfitting
        attention_type: str = "global",  # "global" or "dual" (local+global)
        dual_window_size: int = 8,  # Window size for local attention in dual mode
        dual_edge_mask: float = 0.15,  # Edge mask fraction for global in dual mode
    ):
        """
        Initialize the bad region detector with bottleneck attention.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            use_power_spectrum: If True, expects 2-channel input
            attention_heads: Number of attention heads
            attention_layers: Number of attention layers (1-2 recommended)
            attention_dropout: Dropout rate for attention
            norm_type: Normalization type - "batch", "group", or "instance"
                       IMPORTANT: Use "group" to prevent train/test distribution mismatch!
                       BatchNorm learns statistics from training batches (which have gt_bias sampling).
                       At inference on full images, the distribution differs, causing prediction drift.
                       GroupNorm normalizes within each sample independently - no drift!
            decoder_dropout: Dropout rate for decoder conv blocks (0.0-0.3 recommended).
                            Helps prevent overfitting on long training runs.
                            Set to 0.2-0.3 for 100+ epoch training.
            attention_type: "global" (default, single global attention), "dual" (local+global),
                           or "none" (no attention, conv-only bottleneck for ablation study)
                           DUAL mode addresses edge FP problem by:
                           - Local: 8x8 windowed attention (immune to edge artifacts)
                           - Global: Full attention with edge masking (outer 15% masked)
                           NONE mode replaces attention with two additional conv blocks
                           (roughly same parameter count) for ablation studies.
            dual_window_size: Window size for local attention in dual mode (default 8)
            dual_edge_mask: Fraction of edges to mask in global attention (default 0.15)
        """
        super().__init__()
        self.use_power_spectrum = use_power_spectrum
        self.norm_type = norm_type
        self.decoder_dropout = decoder_dropout
        self.attention_type = attention_type
        
        # Encoder (downsampling path) - same as BadRegionDetector
        self.enc1 = self._conv_block(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = self._conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = self._conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = self._conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck with ATTENTION (or plain conv for ablation)
        self.bottleneck_conv = self._conv_block(512, 1024)
        
        if attention_type == "none":
            # NO ATTENTION: Replace attention with additional conv blocks (for ablation study)
            # Two conv blocks to roughly match the parameter budget of attention
            self.bottleneck_attention = nn.Sequential(
                self._conv_block(1024, 1024),
                self._conv_block(1024, 1024),
            )
        elif attention_type == "dual":
            # DUAL: Local (windowed) + Global (masked) attention
            # Addresses edge FP problem by:
            # - Local: texture discrimination, immune to edges
            # - Global: structure understanding, edges masked out
            self.bottleneck_attention = DualLocalGlobalAttention(
                embed_dim=1024,
                local_heads=attention_heads // 2,  # Half heads for each branch
                global_heads=attention_heads // 2,
                num_layers=attention_layers,
                dropout=attention_dropout,
                window_size=dual_window_size,
                edge_mask_fraction=dual_edge_mask,
            )
        else:
            # GLOBAL: Single global attention (default, original behavior)
            self.bottleneck_attention = BottleneckSelfAttention(
                embed_dim=1024,
                num_heads=attention_heads,
                num_layers=attention_layers,
                dropout=attention_dropout,
            )
        
        # Decoder (upsampling path) - with optional dropout to prevent overfitting
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512, dropout=decoder_dropout)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256, dropout=decoder_dropout)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128, dropout=decoder_dropout)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64, dropout=decoder_dropout)
        
        # Final classification layer (multi-class)
        self.final = nn.Conv2d(64, num_classes, 1)
        
        # Binary badness head
        self.binary_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
    
    def _get_norm_layer(self, num_channels: int) -> nn.Module:
        """Get normalization layer based on norm_type."""
        if self.norm_type == "group":
            # GroupNorm with 8 groups (or fewer if channels < 8)
            num_groups = min(8, num_channels)
            return nn.GroupNorm(num_groups, num_channels)
        elif self.norm_type == "instance":
            return nn.InstanceNorm2d(num_channels, affine=True)
        else:  # "batch" (default)
            return nn.BatchNorm2d(num_channels)
        
    def _conv_block(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> nn.Sequential:
        """Create a convolutional block with normalization, ReLU, and optional dropout."""
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            self._get_norm_layer(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend([
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            self._get_norm_layer(out_channels),
            nn.ReLU(inplace=True),
        ])
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with bottleneck attention."""
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        
        # Bottleneck with ATTENTION (key difference from BadRegionDetector)
        x = self.bottleneck_conv(x)
        x = self.bottleneck_attention(x)  # Global context!
        
        # Decoder with skip connections
        x = self.up4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Final outputs
        multiclass_logits = self.final(x)
        binary_logits = self.binary_head(x)
        
        return multiclass_logits, binary_logits


class SimpleBadRegionDetector(nn.Module):
    """
    Simpler CNN model for bad region detection.
    Good for quick prototyping and smaller datasets.
    
    Can optionally use power spectrum (Fourier space) as additional input channel,
    using complementary real-space and Fourier-space information.
    """
    
    def __init__(self, input_channels: int = 1, use_power_spectrum: bool = False, num_classes: int = 2):
        """
        Initialize the simple bad region detector.
        
        Args:
            input_channels: Number of input channels
            use_power_spectrum: If True, expects 2-channel input
            num_classes: Number of output classes (default: 2)
        
        Args:
            input_channels: Number of input channels (1 for grayscale, 2 for image+power_spectrum)
            use_power_spectrum: If True, expects 2-channel input (image + power spectrum)
        """
        super(SimpleBadRegionDetector, self).__init__()
        self.use_power_spectrum = use_power_spectrum
        self.num_classes = num_classes
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Upsampling to original size
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.num_classes, 1),  # Multi-class classification
        )
        
        # Binary badness head
        self.binary_head = nn.Sequential(
            nn.Conv2d(self.num_classes, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),  # Single channel: probability of badness
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        multiclass_logits = self.upsample(x)
        binary_logits = self.binary_head(multiclass_logits)
        return multiclass_logits, binary_logits


class ResNetUNet(nn.Module):
    """
    ResNet U-Net architecture for pixel-level segmentation.
    Uses ImageNet pretrained ResNet as encoder with U-Net decoder.
    
    Recommended for transfer learning: 
    - ImageNet pretrained ResNet features (proven effective)
    - Fine-tunes well with 10-20 images after base training
    - Supports power-spectrum input
    
    Architecture:
    - ResNet backbone (pretrained on ImageNet)
    - U-Net decoder with skip connections
    - First ResNet layer adapted for 2-channel input if using power spectrum
    """
    
    def __init__(self, input_channels: int = 1, num_classes: int = 2, 
                 resnet_type: str = "resnet18", use_power_spectrum: bool = False,
                 pretrained: bool = True):
        """
        Initialize ResNet U-Net.
        
        Args:
            input_channels: Number of input channels (1 or 2)
            num_classes: Number of output classes
            resnet_type: ResNet variant ("resnet18", "resnet34", "resnet50")
            use_power_spectrum: If True, expects 2-channel input
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(ResNetUNet, self).__init__()
        self.use_power_spectrum = use_power_spectrum
        
        # Load pretrained ResNet backbone
        if resnet_type == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            encoder_channels = [64, 64, 128, 256, 512]  # Initial, layer1, layer2, layer3, layer4
        elif resnet_type == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            encoder_channels = [64, 64, 128, 256, 512]
        elif resnet_type == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            encoder_channels = [64, 256, 512, 1024, 2048]  # ResNet50 uses bottlenecks
        else:
            raise ValueError(f"Unknown ResNet type: {resnet_type}")
        
        # Extract ResNet layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Adapt first layer if input channels don't match ImageNet (3 channels)
        old_conv = self.conv1
        if input_channels != old_conv.in_channels:
            old_conv = self.conv1
            new_conv = nn.Conv2d(input_channels, old_conv.out_channels, 
                               kernel_size=old_conv.kernel_size,
                               stride=old_conv.stride, 
                               padding=old_conv.padding,
                               bias=old_conv.bias is not None)
            
            # Handle weight initialization based on input channels
            with torch.no_grad():
                if pretrained and old_conv.in_channels == 3:
                    # Adapting from ImageNet 3-channel pretrained weights
                    if input_channels == 1:
                        # Single channel: use first channel or average RGB
                        new_conv.weight[:, 0] = old_conv.weight[:, 0]  # Use R channel
                    elif input_channels == 2:
                        # Two channels: copy first channel, initialize second
                        new_conv.weight[:, 0:1] = old_conv.weight[:, 0:1]
                        # Initialize 2nd channel (power spectrum) with Xavier
                        nn.init.xavier_uniform_(new_conv.weight[:, 1:2])
                    else:
                        # More than 3 channels: copy first 3, initialize rest
                        new_conv.weight[:, :3] = old_conv.weight
                        nn.init.xavier_uniform_(new_conv.weight[:, 3:])
                else:
                    # Not pretrained or unexpected input: initialize all weights
                    nn.init.xavier_uniform_(new_conv.weight)
                
                if old_conv.bias is not None:
                    new_conv.bias = old_conv.bias
            
            self.conv1 = new_conv
        
        # Decoder (U-Net style with skip connections)
        # up1: layer4 -> layer3
        self.up1 = nn.ConvTranspose2d(encoder_channels[4], encoder_channels[3], 2, stride=2)
        self.dec1 = self._decoder_block(encoder_channels[3] + encoder_channels[3], encoder_channels[3], encoder_channels[2])
        
        # up2: dec1 -> layer2
        self.up2 = nn.ConvTranspose2d(encoder_channels[2], encoder_channels[2], 2, stride=2)
        self.dec2 = self._decoder_block(encoder_channels[2] + encoder_channels[2], encoder_channels[2], encoder_channels[1])
        
        # up3: dec2 -> layer1
        self.up3 = nn.ConvTranspose2d(encoder_channels[1], encoder_channels[1], 2, stride=2)
        self.dec3 = self._decoder_block(encoder_channels[1] + encoder_channels[1], encoder_channels[1], encoder_channels[0])
        
        # up4: dec3 -> initial (after maxpool)
        self.up4 = nn.ConvTranspose2d(encoder_channels[0], encoder_channels[0], 2, stride=2)
        self.dec4 = self._decoder_block(encoder_channels[0] + encoder_channels[0], encoder_channels[0], encoder_channels[0])
        
        # up5: dec4 -> original input size (512x512)
        # Need one more upsample to get from 256x256 back to 512x512
        self.up5 = nn.ConvTranspose2d(encoder_channels[0], 64, 2, stride=2)
        self.dec5 = self._decoder_block(64 + input_channels, 64, 64)
        
        # Final classification layer (multi-class)
        self.final = nn.Conv2d(64, num_classes, 1)
        
        # Binary badness head (predicts bad vs good as single probability)
        self.binary_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),  # Single channel: probability of badness
        )
    
    def _decoder_block(self, in_channels: int, mid_channels: int, out_channels: int):
        """Create decoder block with conv + batch norm + ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Encoder (ResNet backbone)
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x0)
        
        x1 = self.layer1(x1)  # Skip connection 1
        x2 = self.layer2(x1)  # Skip connection 2
        x3 = self.layer3(x2)  # Skip connection 3
        x4 = self.layer4(x3)  # Bottleneck
        
        # Decoder with skip connections
        up1 = self.up1(x4)
        up1 = torch.cat([up1, x3], dim=1)
        dec1 = self.dec1(up1)
        
        up2 = self.up2(dec1)
        up2 = torch.cat([up2, x2], dim=1)
        dec2 = self.dec2(up2)
        
        up3 = self.up3(dec2)
        up3 = torch.cat([up3, x1], dim=1)
        dec3 = self.dec3(up3)
        
        up4 = self.up4(dec3)
        up4 = torch.cat([up4, x0], dim=1)
        dec4 = self.dec4(up4)
        
        # One more upsample to get back to original input size (512x512)
        up5 = self.up5(dec4)
        # Use original input as skip connection (upsampled if needed for size matching)
        if up5.shape[2:] != x.shape[2:]:
            up5 = F.interpolate(up5, size=x.shape[2:], mode='bilinear', align_corners=False)
        # Concatenate with original input as final skip connection
        dec5 = self.dec5(torch.cat([up5, x], dim=1))
        
        # Final outputs
        multiclass_logits = self.final(dec5)
        binary_logits = self.binary_head(dec5)
        
        return multiclass_logits, binary_logits


class EdgeDirectionWrapper(nn.Module):
    """
    Wrapper that adds directional edge features (3 scalars) to any segmentation model.
    
    Much simpler than GlobalFeatureWrapper - uses only 3 scalars:
    1. direction_strength: How aligned are gradients? (carbon edge = high)
    2. direction_angle: Which way does the edge run? (normalized to [-1, 1])
    3. edge_proximity: How close is THIS patch to the edge? (patch-specific!)
    
    The key innovation is that edge_proximity is DIFFERENT for each patch,
    providing spatial localization that the old global features lacked.
    """
    
    def __init__(self, base_model: nn.Module, output_dim: int = 64, num_classes: int = 1):
        super().__init__()
        self.base_model = base_model
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        # Small MLP: 3 scalars → 32 → 64-dim embedding
        self.edge_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, output_dim),
            nn.ReLU(inplace=True),
        )
        
        # Blend layer: concatenate edge features with base logits, then blend
        self.blend_conv = nn.Conv2d(num_classes + output_dim, num_classes, kernel_size=1, bias=True)
        
        # Initialize to mostly pass through original predictions
        nn.init.zeros_(self.blend_conv.weight)
        nn.init.zeros_(self.blend_conv.bias)
        for i in range(num_classes):
            self.blend_conv.weight.data[i, i, 0, 0] = 1.0
    
    def forward(self, x: torch.Tensor, edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional edge features.
        
        Args:
            x: Input patches, shape [B, C, H, W]
            edge_features: Edge features, shape [B, 3] (optional)
            
        Returns:
            Logits, shape [B, num_classes, H, W]
        """
        # Get base model predictions
        base_out = self.base_model(x)
        
        if isinstance(base_out, (tuple, list)):
            base_logits = base_out[0]
        else:
            base_logits = base_out
        
        # If no edge features, return base predictions
        if edge_features is None:
            return base_logits
        
        B, C, H, W = base_logits.shape
        
        # Process edge features through MLP
        edge_embedding = self.edge_mlp(edge_features)  # [B, output_dim]
        
        # Broadcast spatially: [B, output_dim] → [B, output_dim, H, W]
        edge_spatial = edge_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        
        # Concatenate with base logits
        combined = torch.cat([base_logits, edge_spatial], dim=1)  # [B, C+output_dim, H, W]
        
        # Blend to get final logits
        final_logits = self.blend_conv(combined)  # [B, C, H, W]
        
        return final_logits


class GlobalFeatureWrapper(nn.Module):
    """
    Wrapper that adds global feature support to any segmentation model.
    
    Global features (154-dim) capture micrograph-wide characteristics that help
    classify patches in uniform regions (like the middle of large carbon areas)
    that lack local distinguishing features.
    
    Architecture:
        Base model: patch → encoder → decoder → logits
        With global: 
            patch → encoder → features
            global_features → MLP → embedding (spatially broadcast)
            concat(features, embedding) → adjusted decoder → logits
    """
    
    def __init__(self, base_model: nn.Module, global_feature_dim: int = 154, 
                 hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.base_model = base_model
        self.global_feature_dim = global_feature_dim
        self.num_classes = num_classes
        
        # Global feature MLP: 154 → 256 → 256 → 64
        # Output 64-dim to concatenate with decoder features without too much overhead
        self.global_mlp = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
        )
        
        # Adjustment layer: takes concatenated features and adjusts to original channel count
        # This is applied after the base model's last decoder layer
        # We'll use a simple 1x1 conv to blend global info with local predictions
        self.blend_conv = nn.Conv2d(num_classes + 64, num_classes, kernel_size=1, bias=True)
        
        # Initialize blend_conv to mostly pass through original predictions
        nn.init.zeros_(self.blend_conv.weight)
        nn.init.zeros_(self.blend_conv.bias)
        # Identity-like initialization for first num_classes channels
        for i in range(num_classes):
            self.blend_conv.weight.data[i, i, 0, 0] = 1.0
    
    def forward(self, x: torch.Tensor, global_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional global features.
        
        Args:
            x: Input patches, shape [B, C, H, W]
            global_features: Global features, shape [B, 154] (optional)
            
        Returns:
            Logits, shape [B, num_classes, H, W]
        """
        # Get base model predictions
        base_out = self.base_model(x)
        
        # Handle tuple outputs (some models return multiple tensors)
        if isinstance(base_out, (tuple, list)):
            base_logits = base_out[0]
        else:
            base_logits = base_out
        
        # If no global features, return base predictions
        if global_features is None:
            return base_logits
        
        B, C, H, W = base_logits.shape
        
        # Process global features
        global_embedding = self.global_mlp(global_features)  # [B, 64]
        
        # Broadcast spatially: [B, 64] → [B, 64, H, W]
        global_spatial = global_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        
        # Concatenate with base logits
        combined = torch.cat([base_logits, global_spatial], dim=1)  # [B, C+64, H, W]
        
        # Blend to get final logits
        final_logits = self.blend_conv(combined)  # [B, C, H, W]
        
        return final_logits


class ViTContaminationDetector(nn.Module):
    """
    Contamination detector with Vision Transformer for full-image context.
    
    Architecture:
        patches [B, N, 2, H, W]
            ↓
        CNN encoder (per-patch, parallel)
            ↓
        patch embeddings [B, N, embed_dim]
            ↓
        Vision Transformer (self-attention across all N patches)
            ↓
        contextualized embeddings [B, N, embed_dim]
            ↓
        CNN decoder (per-patch, parallel)
            ↓
        predictions [B, N, 1, H, W]
    
    The ViT enables patches to communicate via self-attention, allowing patches
    in uniform regions (like carbon interiors) to learn context from neighbors.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        embed_dim: int = 512,  # Dimension of patch embedding from encoder bottleneck
        vit_depth: int = 6,
        vit_heads: int = 8,
        vit_dropout: float = 0.1,
    ):
        super().__init__()
        
        # Import ViT module
        from .vit import PatchViT
        
        self.base_model = base_model
        self.embed_dim = embed_dim
        
        # Vision Transformer for patch contextualization
        self.vit = PatchViT(
            patch_embed_dim=embed_dim,
            num_heads=vit_heads,
            depth=vit_depth,
            dropout=vit_dropout,
        )
        
        # We need to hook into the base model to extract embeddings
        # The bottleneck is after the encoder and before the decoder.
        self.use_embedding_hook = False
        self._embedding = None
        self._hook_handle = None
        
    def _embedding_hook(self, module, input, output):
        """Hook to capture bottleneck embeddings."""
        self._embedding = output
    
    def forward(
        self,
        patches: torch.Tensor,
        grid_shape: Optional[tuple] = None,
        use_vit: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with optional ViT contextualization.
        
        Args:
            patches: [B, N, C, H, W] all patches from B micrographs
            grid_shape: (grid_h, grid_w) for positional embeddings
            use_vit: If False, process patches independently (baseline behavior)
            
        Returns:
            predictions: [B, N, 1, H, W] predictions for all patches
        """
        B, N, C, H, W = patches.shape
        
        if not use_vit:
            # Baseline: process each patch independently
            patches_flat = patches.view(B * N, C, H, W)
            out = self.base_model(patches_flat)
            if isinstance(out, (tuple, list)):
                out = out[0]
            # Handle both (B*N, 1, H, W) and (B*N, H, W) shapes
            if out.dim() == 3:
                out = out.unsqueeze(1)
            return out.view(B, N, 1, H // 4, W // 4)  # Assuming 4x downscale
        
        # ViT path: need to extract embeddings, apply ViT, then decode
        # This requires a specialized architecture that exposes the bottleneck
        # For now, use a simpler approach: compute full outputs, pool to embedding,
        # apply ViT, then modulate outputs
        
        # Process all patches through base model
        patches_flat = patches.view(B * N, C, H, W)
        base_out = self.base_model(patches_flat)
        if isinstance(base_out, (tuple, list)):
            base_logits = base_out[0]
        else:
            base_logits = base_out
        
        # Handle output dimensions
        if base_logits.dim() == 3:
            base_logits = base_logits.unsqueeze(1)
        
        _, out_C, out_H, out_W = base_logits.shape
        
        # Create embedding from spatial average of logits
        # This is a simple approach; more sophisticated would hook into encoder
        patch_embeds = base_logits.mean(dim=(2, 3))  # [B*N, out_C]
        
        # Pad to embed_dim if needed
        if patch_embeds.shape[-1] < self.embed_dim:
            pad = torch.zeros(B * N, self.embed_dim - patch_embeds.shape[-1], 
                            device=patch_embeds.device)
            patch_embeds = torch.cat([patch_embeds, pad], dim=-1)
        
        # Reshape to [B, N, embed_dim]
        patch_embeds = patch_embeds.view(B, N, -1)
        
        # Apply ViT
        contextualized = self.vit(patch_embeds, grid_shape)  # [B, N, embed_dim]
        
        # Simple modulation: scale base predictions by ViT output
        # (more sophisticated: use contextualized features to generate modulation weights)
        modulation = contextualized[:, :, :out_C].view(B * N, out_C, 1, 1)
        modulation = torch.sigmoid(modulation)  # Scale factor [0, 1]
        
        # Apply modulation
        modulated_logits = base_logits * (0.5 + modulation)  # Soft modulation
        
        # Reshape back
        return modulated_logits.view(B, N, out_C, out_H, out_W)


class ViTEncoderDecoderWrapper(nn.Module):
    """
    Wrapper that adds ViT between encoder and decoder stages.
    
    This properly integrates ViT by:
    1. Running encoder on all patches (parallel)
    2. Extracting bottleneck features
    3. Applying ViT self-attention across patches
    4. Running decoder on contextualized features
    
    This is the recommended architecture for ViT integration.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        vit_depth: int = 6,
        vit_heads: int = 8,
        vit_dropout: float = 0.1,
        input_channels: int = 2,
        num_classes: int = 1,
    ):
        super().__init__()
        
        from .vit import PatchViT
        
        # Encoder: process individual patches to get embeddings
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # [B, 512, 4, 4]
        )
        
        # Project to ViT embedding dimension
        self.embed_proj = nn.Linear(512 * 4 * 4, embed_dim)
        
        # Vision Transformer
        self.vit = PatchViT(
            patch_embed_dim=embed_dim,
            num_heads=vit_heads,
            depth=vit_depth,
            dropout=vit_dropout,
        )
        
        # Project back for decoder
        self.decode_proj = nn.Linear(embed_dim, 512 * 4 * 4)
        
        # Decoder: reconstruct spatial predictions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, num_classes, 1),
        )
        
        self.embed_dim = embed_dim
    
    def forward(
        self,
        patches: torch.Tensor,
        grid_shape: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Forward pass with ViT contextualization.
        
        Args:
            patches: [B, N, C, H, W] all patches from B micrographs
            grid_shape: (grid_h, grid_w) for positional embeddings
            
        Returns:
            predictions: [B, N, num_classes, out_H, out_W]
        """
        B, N, C, H, W = patches.shape
        
        # Flatten batch and patches for parallel encoding
        patches_flat = patches.view(B * N, C, H, W)
        
        # Encode
        enc_features = self.encoder(patches_flat)  # [B*N, 512, 4, 4]
        enc_flat = enc_features.view(B * N, -1)  # [B*N, 512*4*4]
        embeddings = self.embed_proj(enc_flat)  # [B*N, embed_dim]
        
        # Reshape for ViT
        embeddings = embeddings.view(B, N, -1)  # [B, N, embed_dim]
        
        # Apply ViT
        contextualized = self.vit(embeddings, grid_shape)  # [B, N, embed_dim]
        
        # Reshape back for decoding
        contextualized = contextualized.view(B * N, -1)  # [B*N, embed_dim]
        decode_features = self.decode_proj(contextualized)  # [B*N, 512*4*4]
        decode_features = decode_features.view(B * N, 512, 4, 4)
        
        # Decode
        predictions = self.decoder(decode_features)  # [B*N, num_classes, out_H, out_W]
        
        # Reshape to [B, N, ...]
        _, out_C, out_H, out_W = predictions.shape
        return predictions.view(B, N, out_C, out_H, out_W)


def create_vit_model(
    embed_dim: int = 512,
    vit_depth: int = 6,
    vit_heads: int = 8,
    vit_dropout: float = 0.1,
    input_channels: int = 2,
    num_classes: int = 1,
    device: str = "cuda",
) -> nn.Module:
    """
    Create a ViT-based contamination detector.
    
    Args:
        embed_dim: Embedding dimension for ViT
        vit_depth: Number of transformer layers
        vit_heads: Number of attention heads
        vit_dropout: Dropout rate
        input_channels: Input channels (2 for real + PSD)
        num_classes: Number of output classes
        device: Target device
        
    Returns:
        ViT model on specified device
    """
    model = ViTEncoderDecoderWrapper(
        embed_dim=embed_dim,
        vit_depth=vit_depth,
        vit_heads=vit_heads,
        vit_dropout=vit_dropout,
        input_channels=input_channels,
        num_classes=num_classes,
    )
    return model.to(device)


def create_model(model_type: str = "unet_attention", device: str = "cuda", use_power_spectrum: bool = True,
                 resnet_type: str = "resnet18", pretrained: bool = True,
                 num_classes: int = 2,
                 use_global_features: bool = False,
                 global_feature_dim: int = 154,
                 use_edge_direction: bool = False,
                 edge_direction_dim: int = 64,
                 norm_type: str = "batch",
                 decoder_dropout: float = 0.0,
                 attention_type: str = "global",
                 dual_window_size: int = 8,
                 dual_edge_mask: float = 0.15,
                 input_channels_override: Optional[int] = None) -> nn.Module:
    """
    Create a bad region detector model.
    
    Args:
        model_type: Type of model ("simple", "unet", "unet_attention", or "resnet_unet")
                   Default: "unet_attention"
        device: Device to place model on
        use_power_spectrum: If True, model expects 2-channel input (image + power spectrum)
                          Default: True
        resnet_type: ResNet variant for resnet_unet ("resnet18", "resnet34", "resnet50")
        pretrained: Whether to use ImageNet pretrained weights (for resnet_unet)
        use_global_features: If True, wrap model with GlobalFeatureWrapper for global context
        global_feature_dim: Dimension of global features (default: 154)
        use_edge_direction: If True, wrap model with EdgeDirectionWrapper (3 scalars, recommended)
        edge_direction_dim: Output dimension of edge MLP (default: 64)
        attention_type: "global" (single global attention), "dual" (local+global attention),
                       or "none" (conv-only bottleneck for ablation study).
                       DUAL mode helps eliminate edge FPs by using windowed local attention
                       (immune to edges) + masked global attention (outer 15% masked out)
        dual_window_size: Window size for local attention in dual mode (default 8)
        dual_edge_mask: Fraction of edges to mask in global attention (default 0.15)
        
    Returns:
        Initialized model
    """
    input_channels = int(input_channels_override) if input_channels_override is not None else (2 if use_power_spectrum else 1)
    
    if model_type == "simple":
        model = SimpleBadRegionDetector(input_channels=input_channels, use_power_spectrum=use_power_spectrum, num_classes=num_classes)
    elif model_type == "unet":
        model = BadRegionDetector(input_channels=input_channels, num_classes=num_classes, use_power_spectrum=use_power_spectrum)
    elif model_type == "resnet_unet":
        model = ResNetUNet(input_channels=input_channels, num_classes=num_classes, 
                          resnet_type=resnet_type, use_power_spectrum=use_power_spectrum,
                          pretrained=pretrained)
    elif model_type == "unet_attention":
        # U-Net with self-attention at bottleneck for global context
        # Best of both worlds: CNN inductive bias + global reasoning at coarse scale
        model = BadRegionDetectorWithAttention(
            input_channels=input_channels, 
            num_classes=num_classes, 
            use_power_spectrum=use_power_spectrum,
            attention_heads=8,
            attention_layers=2,
            attention_dropout=0.1,
            norm_type=norm_type,
            decoder_dropout=decoder_dropout,
            attention_type=attention_type,
            dual_window_size=dual_window_size,
            dual_edge_mask=dual_edge_mask,
        )
        norm_str = "GroupNorm (recommended)" if norm_type == "group" else f"{norm_type.capitalize()}Norm"
        dropout_str = f", decoder_dropout={decoder_dropout}" if decoder_dropout > 0 else ""
        if attention_type == "dual":
            attn_str = f"DUAL local+global (window={dual_window_size}, edge_mask={dual_edge_mask})"
        elif attention_type == "none":
            attn_str = "NONE (conv-only bottleneck, for ablation)"
        else:
            attn_str = "global (8 heads)"
        print(f"  ✓ Using U-Net with {attn_str} attention, 2 layers, {norm_str}{dropout_str}")
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: simple, unet, unet_attention, resnet_unet")
    
    # Wrap with EdgeDirectionWrapper if enabled (preferred over GlobalFeatureWrapper)
    if use_edge_direction:
        model = EdgeDirectionWrapper(model, output_dim=edge_direction_dim, num_classes=num_classes)
        print(f"  ✓ Wrapped model with EdgeDirectionWrapper (3 scalars → {edge_direction_dim}-dim)")
    elif use_global_features:
        model = GlobalFeatureWrapper(model, global_feature_dim=global_feature_dim, num_classes=num_classes)
        print(f"  ✓ Wrapped model with GlobalFeatureWrapper (input dim={global_feature_dim})")
    
    model = model.to(device)
    return model
