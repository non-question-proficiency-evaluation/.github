"""
Unit tests for data loading functionality.

Tests cover DATA and PID_DATA classes and their load_data methods.
"""
import unittest
import numpy as np
import sys
import os
import tempfile

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from load_data import DATA, PID_DATA


class TestDataLoading(unittest.TestCase):
    """
    Test cases for data loading classes.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.n_question = 10
        self.seqlen = 20
        self.separate_char = ','
        
    def create_test_data_file(self, format_type='cid'):
        """Create a temporary test data file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        
        if format_type == 'cid':
            # 3-line format
            temp_file.write("1,1000\n")  # Student ID, timestamp
            temp_file.write("1,2,3,4,5\n")  # Question IDs
            temp_file.write("1,0,1,1,0\n")  # Answers
            temp_file.write("2,2000\n")
            temp_file.write("2,3,4,5,6\n")
            temp_file.write("0,1,0,1,1\n")
        else:  # pid format
            # 4-line format
            temp_file.write("1,1000\n")  # Student ID, timestamp
            temp_file.write("101,102,103,104,105\n")  # Problem IDs
            temp_file.write("1,2,3,4,5\n")  # Question IDs
            temp_file.write("1,0,1,1,0\n")  # Answers
            temp_file.write("2,2000\n")
            temp_file.write("106,107,108,109,110\n")
            temp_file.write("2,3,4,5,6\n")
            temp_file.write("0,1,0,1,1\n")
        
        temp_file.close()
        return temp_file.name
    
    def test_data_init(self):
        """Test DATA class initialization."""
        loader = DATA(
            n_question=self.n_question,
            seqlen=self.seqlen,
            separate_char=self.separate_char
        )
        self.assertEqual(loader.n_question, self.n_question)
        self.assertEqual(loader.seqlen, self.seqlen)
        self.assertEqual(loader.separate_char, self.separate_char)
        
    def test_pid_data_init(self):
        """Test PID_DATA class initialization."""
        loader = PID_DATA(
            n_question=self.n_question,
            seqlen=self.seqlen,
            separate_char=self.separate_char
        )
        self.assertEqual(loader.n_question, self.n_question)
        self.assertEqual(loader.seqlen, self.seqlen)
        
    def test_data_load_cid(self):
        """Test loading CID format data."""
        test_file = self.create_test_data_file('cid')
        loader = DATA(
            n_question=self.n_question,
            seqlen=self.seqlen,
            separate_char=self.separate_char
        )
        
        try:
            q_data, qa_data, student_ids = loader.load_data(test_file)
            
            self.assertIsInstance(q_data, np.ndarray)
            self.assertIsInstance(qa_data, np.ndarray)
            self.assertIsInstance(student_ids, np.ndarray)
            self.assertEqual(q_data.shape[1], self.seqlen)
            self.assertEqual(qa_data.shape[1], self.seqlen)
        finally:
            os.unlink(test_file)
            
    def test_pid_data_load(self):
        """Test loading PID format data."""
        test_file = self.create_test_data_file('pid')
        loader = PID_DATA(
            n_question=self.n_question,
            seqlen=self.seqlen,
            separate_char=self.separate_char
        )
        
        try:
            q_data, qa_data, pid_data = loader.load_data(test_file)
            
            self.assertIsInstance(q_data, np.ndarray)
            self.assertIsInstance(qa_data, np.ndarray)
            self.assertIsInstance(pid_data, np.ndarray)
            self.assertEqual(q_data.shape[1], self.seqlen)
            self.assertEqual(qa_data.shape[1], self.seqlen)
            self.assertEqual(pid_data.shape[1], self.seqlen)
        finally:
            os.unlink(test_file)
            
    def test_data_load_file_not_found(self):
        """Test error handling for missing file."""
        loader = DATA(
            n_question=self.n_question,
            seqlen=self.seqlen,
            separate_char=self.separate_char
        )
        
        with self.assertRaises(FileNotFoundError):
            loader.load_data('nonexistent_file.csv')


if __name__ == '__main__':
    unittest.main()
