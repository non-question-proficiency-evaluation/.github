"""
Unit tests for AKT model implementation.

Tests cover model initialization, forward pass, and basic functionality.
"""
import unittest
import torch
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from akt import AKT


class TestAKT(unittest.TestCase):
    """
    Test cases for AKT model class.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.n_question = 110
        self.n_pid = 100
        self.d_model = 256
        self.n_blocks = 1
        self.kq_same = 1
        self.dropout = 0.05
        self.model_type = 'akt'
        self.batch_size = 4
        self.seq_len = 50
        
    def test_akt_init_pid(self):
        """Test AKT model initialization with Problem IDs (Rasch model)."""
        model = AKT(
            n_question=self.n_question,
            n_pid=self.n_pid,
            d_model=self.d_model,
            n_blocks=self.n_blocks,
            kq_same=self.kq_same,
            dropout=self.dropout,
            model_type=self.model_type
        )
        self.assertIsNotNone(model)
        self.assertEqual(model.n_question, self.n_question)
        self.assertEqual(model.n_pid, self.n_pid)
        
    def test_akt_init_cid(self):
        """Test AKT model initialization without Problem IDs (NonRasch model)."""
        model = AKT(
            n_question=self.n_question,
            n_pid=0,  # CID model
            d_model=self.d_model,
            n_blocks=self.n_blocks,
            kq_same=self.kq_same,
            dropout=self.dropout,
            model_type=self.model_type
        )
        self.assertIsNotNone(model)
        self.assertEqual(model.n_pid, 0)
        
    def test_akt_forward_pid(self):
        """Test AKT forward pass with Problem IDs."""
        model = AKT(
            n_question=self.n_question,
            n_pid=self.n_pid,
            d_model=self.d_model,
            n_blocks=self.n_blocks,
            kq_same=self.kq_same,
            dropout=self.dropout,
            model_type=self.model_type
        )
        model.eval()
        
        q_data = torch.randint(0, self.n_question, (self.batch_size, self.seq_len))
        qa_data = torch.randint(0, 2 * self.n_question, (self.batch_size, self.seq_len))
        target = torch.randint(0, 2, (self.batch_size, self.seq_len)).float()
        pid_data = torch.randint(0, self.n_pid, (self.batch_size, self.seq_len))
        
        with torch.no_grad():
            loss, pred, mask_sum = model(q_data, qa_data, target, pid_data)
        
        self.assertIsNotNone(loss)
        self.assertEqual(pred.shape[0], self.batch_size * self.seq_len)
        self.assertGreater(mask_sum.item(), 0)
        
    def test_akt_forward_cid(self):
        """Test AKT forward pass without Problem IDs."""
        model = AKT(
            n_question=self.n_question,
            n_pid=0,
            d_model=self.d_model,
            n_blocks=self.n_blocks,
            kq_same=self.kq_same,
            dropout=self.dropout,
            model_type=self.model_type
        )
        model.eval()
        
        q_data = torch.randint(0, self.n_question, (self.batch_size, self.seq_len))
        qa_data = torch.randint(0, 2 * self.n_question, (self.batch_size, self.seq_len))
        target = torch.randint(0, 2, (self.batch_size, self.seq_len)).float()
        
        with torch.no_grad():
            loss, pred, mask_sum = model(q_data, qa_data, target)
        
        self.assertIsNotNone(loss)
        self.assertEqual(pred.shape[0], self.batch_size * self.seq_len)
        
    def test_akt_reset(self):
        """Test difficulty parameter reset for Rasch models."""
        model = AKT(
            n_question=self.n_question,
            n_pid=self.n_pid,
            d_model=self.d_model,
            n_blocks=self.n_blocks,
            kq_same=self.kq_same,
            dropout=self.dropout,
            model_type=self.model_type
        )
        # Reset should not raise an error
        model.reset()
        self.assertTrue(True)  # If we get here, reset worked


if __name__ == '__main__':
    unittest.main()
