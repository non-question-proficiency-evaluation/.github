"""
Unit tests for utility functions.

Tests cover model loading, file name generation, and other utilities.
"""
import unittest
import sys
import os
from types import SimpleNamespace

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import model_isPid_type, get_file_name_identifier, try_makedirs


class TestUtils(unittest.TestCase):
    """
    Test cases for utility functions.
    """
    
    def test_model_isPid_type_pid(self):
        """Test model_isPid_type with PID model."""
        is_pid, model_type = model_isPid_type('akt_pid')
        self.assertTrue(is_pid)
        self.assertEqual(model_type, 'akt')
        
    def test_model_isPid_type_cid(self):
        """Test model_isPid_type with CID model."""
        is_pid, model_type = model_isPid_type('akt_cid')
        self.assertFalse(is_pid)
        self.assertEqual(model_type, 'akt')
        
    def test_get_file_name_identifier_akt(self):
        """Test file name identifier generation for AKT model."""
        params = SimpleNamespace(
            model='akt_pid',
            batch_size=24,
            n_block=1,
            maxgradnorm=-1,
            lr=1e-5,
            seed=224,
            seqlen=200,
            dropout=0.05,
            d_model=256,
            train_set=1,
            kq_same=1,
            l2=1e-5
        )
        identifier = get_file_name_identifier(params)
        self.assertIsInstance(identifier, list)
        self.assertGreater(len(identifier), 0)
        
    def test_try_makedirs(self):
        """Test directory creation utility."""
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        test_path = os.path.join(temp_dir, 'test', 'nested', 'dir')
        
        try:
            try_makedirs(test_path)
            self.assertTrue(os.path.exists(test_path))
            self.assertTrue(os.path.isdir(test_path))
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
