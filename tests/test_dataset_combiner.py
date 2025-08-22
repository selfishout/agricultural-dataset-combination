"""
Tests for the dataset combiner module.
"""

import unittest
import tempfile
import os
import shutil
from unittest.mock import Mock, patch

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset_combiner import DatasetCombiner


class TestDatasetCombiner(unittest.TestCase):
    """Test cases for DatasetCombiner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'datasets': {
                'test_dataset': {
                    'source_path': '/test/path',
                    'name': 'Test Dataset'
                }
            },
            'storage': {
                'output_dir': os.path.join(self.temp_dir, 'output'),
                'intermediate_dir': os.path.join(self.temp_dir, 'intermediate')
            },
            'splits': {
                'train': 0.7,
                'validation': 0.2,
                'test': 0.1
            },
            'output': {
                'target_image_size': [512, 512],
                'output_format': 'png',
                'compression_quality': 95
            },
            'processing': {
                'normalization': True,
                'augmentation': True,
                'resize_method': 'bilinear',
                'min_image_size': [256, 256],
                'max_image_size': [2048, 2048],
                'min_annotation_quality': 0.7,
                'augmentation_params': {
                    'horizontal_flip': True,
                    'rotation_range': [-15, 15]
                }
            },
            'annotation_mapping': {
                'classes': {
                    0: 'background',
                    1: 'crop',
                    2: 'weed'
                },
                'test_dataset': {
                    0: 0,
                    1: 1,
                    2: 2
                }
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('dataset_combiner.load_all_datasets')
    @patch('dataset_combiner.Preprocessor')
    @patch('dataset_combiner.AnnotationMapper')
    @patch('dataset_combiner.QualityController')
    def test_initialization(self, mock_qc, mock_am, mock_prep, mock_load):
        """Test DatasetCombiner initialization."""
        # Mock the dataset loaders
        mock_load.return_value = {}
        
        # Mock the components
        mock_prep.return_value = Mock()
        mock_am.return_value = Mock()
        mock_qc.return_value = Mock()
        
        # Create instance
        combiner = DatasetCombiner(self.config)
        
        # Check that components were initialized
        mock_prep.assert_called_once_with(self.config)
        mock_am.assert_called_once_with(self.config)
        mock_qc.assert_called_once_with(self.config)
        
        # Check that directories were created
        self.assertTrue(os.path.exists(self.config['storage']['output_dir']))
        self.assertTrue(os.path.exists(self.config['storage']['intermediate_dir']))
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test with missing required keys
        invalid_config = {}
        
        with self.assertRaises(KeyError):
            DatasetCombiner(invalid_config)
    
    def test_directory_creation(self):
        """Test that output directories are created."""
        with patch('dataset_combiner.load_all_datasets') as mock_load:
            mock_load.return_value = {}
            
            with patch('dataset_combiner.Preprocessor') as mock_prep:
                with patch('dataset_combiner.AnnotationMapper') as mock_am:
                    with patch('dataset_combiner.QualityController') as mock_qc:
                        mock_prep.return_value = Mock()
                        mock_am.return_value = Mock()
                        mock_qc.return_value = Mock()
                        
                        combiner = DatasetCombiner(self.config)
                        
                        # Check directories exist
                        self.assertTrue(os.path.exists(combiner.output_dir))
                        self.assertTrue(os.path.exists(combiner.intermediate_dir))


if __name__ == '__main__':
    unittest.main()
