"""
Tests for Vanguard-FEDformer models.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from vanguard_fedformer.core.models.fedformer import VanguardFEDformer
from vanguard_fedformer.core.models.flows import NormalizingFlow
from vanguard_fedformer.core.models.attention import FourierAttention, WaveletAttention

class TestVanguardFEDformer(unittest.TestCase):
    """Test cases for the main VanguardFEDformer model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.sequence_length = 96
        self.prediction_length = 24
        self.d_model = 64
        self.n_features = 5
        
        self.model = VanguardFEDformer(
            d_model=self.d_model,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            dropout=0.1,
            activation="relu"
        )
        
        self.x = torch.randn(self.batch_size, self.sequence_length, self.n_features)
        self.y = torch.randn(self.batch_size, self.prediction_length, self.n_features)
    
    def test_model_creation(self):
        """Test that the model can be created successfully."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.d_model, self.d_model)
    
    def test_forward_pass(self):
        """Test that the model can perform a forward pass."""
        try:
            output = self.model(self.x)
            self.assertEqual(output.shape, (self.batch_size, self.prediction_length, self.n_features))
        except Exception as e:
            self.fail(f"Forward pass failed: {e}")
    
    def test_model_parameters(self):
        """Test that the model has trainable parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.assertGreater(total_params, 0)
        self.assertGreater(trainable_params, 0)
        self.assertEqual(total_params, trainable_params)

class TestNormalizingFlow(unittest.TestCase):
    """Test cases for normalizing flows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.sequence_length = 96
        self.n_features = 5
        
        self.flow = NormalizingFlow(
            n_flows=2,
            hidden_dim=32,
            flow_type="real_nvp"
        )
        
        self.x = torch.randn(self.batch_size, self.sequence_length, self.n_features)
    
    def test_flow_creation(self):
        """Test that the flow can be created successfully."""
        self.assertIsNotNone(self.flow)
    
    def test_flow_forward(self):
        """Test that the flow can perform a forward pass."""
        try:
            z, log_det = self.flow(self.x)
            self.assertEqual(z.shape, self.x.shape)
            self.assertEqual(log_det.shape, (self.batch_size,))
        except Exception as e:
            self.fail(f"Flow forward pass failed: {e}")

class TestAttention(unittest.TestCase):
    """Test cases for attention mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.sequence_length = 96
        self.d_model = 64
        
        self.fourier_attention = FourierAttention(d_model=self.d_model)
        self.wavelet_attention = WaveletAttention(d_model=self.d_model)
        
        self.x = torch.randn(self.batch_size, self.sequence_length, self.d_model)
    
    def test_fourier_attention(self):
        """Test Fourier attention mechanism."""
        try:
            output = self.fourier_attention(self.x)
            self.assertEqual(output.shape, self.x.shape)
        except Exception as e:
            self.fail(f"Fourier attention failed: {e}")
    
    def test_wavelet_attention(self):
        """Test wavelet attention mechanism."""
        try:
            output = self.wavelet_attention(self.x)
            self.assertEqual(output.shape, self.x.shape)
        except Exception as e:
            self.fail(f"Wavelet attention failed: {e}")

if __name__ == "__main__":
    unittest.main()