"""Unit tests for optimization_manager module."""

import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import pytest

from utils.optimization_manager import (
    get_system_memory_gb,
    get_optimized_settings,
    find_kb_configs,
    show_optimization_tiers,
    process_optimize
)


class TestOptimizationManager:
    """Test optimization manager functionality."""
    
    def test_get_system_memory_gb_with_psutil(self):
        """Test memory detection with psutil available."""
        with patch('psutil.virtual_memory') as mock_vm:
            mock_vm.return_value = Mock(total=17179869184)  # 16GB
            result = get_system_memory_gb()
            assert result == 16.0
    
    def test_get_system_memory_gb_fallback(self):
        """Test memory detection fallback to /proc/meminfo."""
        with patch('psutil.virtual_memory', side_effect=ImportError):
            with patch('builtins.open', Mock(return_value=[
                'MemTotal:       16777216 kB\n',
                'MemFree:        1234567 kB\n'
            ])):
                result = get_system_memory_gb()
                assert result == 16.0
    
    def test_get_optimized_settings_tiers(self):
        """Test optimization settings for different memory tiers."""
        # Low memory tier
        settings = get_optimized_settings(8)
        assert settings['tier'] == 'low'
        assert int(settings['optimizations']['LIMITS']['memory_cache_size']) == 50000
        
        # Medium memory tier
        settings = get_optimized_settings(32)
        assert settings['tier'] == 'medium'
        assert int(settings['optimizations']['LIMITS']['memory_cache_size']) == 100000
        
        # High memory tier
        settings = get_optimized_settings(96)
        assert settings['tier'] == 'high'
        assert int(settings['optimizations']['LIMITS']['memory_cache_size']) == 150000
        
        # Very high memory tier
        settings = get_optimized_settings(256)
        assert settings['tier'] == 'very_high'
        assert int(settings['optimizations']['LIMITS']['memory_cache_size']) == 200000
    
    def test_find_kb_configs(self):
        """Test finding KB configuration files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test structure
            kb1_dir = os.path.join(tmpdir, 'kb1')
            kb2_dir = os.path.join(tmpdir, 'kb2')
            backup_dir = os.path.join(tmpdir, 'kb1', 'backups')
            
            os.makedirs(kb1_dir)
            os.makedirs(kb2_dir)
            os.makedirs(backup_dir)
            
            # Create config files
            with open(os.path.join(kb1_dir, 'kb1.cfg'), 'w') as f:
                f.write('[DEFAULT]\n')
            with open(os.path.join(kb2_dir, 'kb2.cfg'), 'w') as f:
                f.write('[DEFAULT]\n')
            # Should be skipped
            with open(os.path.join(backup_dir, 'kb1.cfg.backup'), 'w') as f:
                f.write('[DEFAULT]\n')
            
            configs = find_kb_configs(tmpdir)
            assert len(configs) == 2
            assert any('kb1.cfg' in c for c in configs)
            assert any('kb2.cfg' in c for c in configs)
            assert not any('backup' in c for c in configs)
    
    def test_show_optimization_tiers(self):
        """Test optimization tier display."""
        result = show_optimization_tiers()
        assert 'Optimization Tiers for CustomKB' in result
        assert 'Low Memory (<16GB)' in result
        assert 'Medium Memory (16-64GB)' in result
        assert 'High Memory (64-128GB)' in result
        assert 'Very High Memory (>128GB)' in result
    
    def test_process_optimize_show_tiers(self):
        """Test process_optimize with --show-tiers flag."""
        args = Mock(show_tiers=True)
        logger = Mock()
        
        result = process_optimize(args, logger)
        assert 'Optimization Tiers for CustomKB' in result
    
    def test_process_optimize_target_resolution(self):
        """Test target resolution in process_optimize."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a config file
            config_path = os.path.join(tmpdir, 'test.cfg')
            with open(config_path, 'w') as f:
                f.write('[DEFAULT]\n')
            
            args = Mock(
                show_tiers=False,
                target=config_path,
                dry_run=True,
                analyze=False,
                memory_gb=None
            )
            logger = Mock()
            
            with patch('utils.optimization_manager.optimize_config') as mock_optimize:
                mock_optimize.return_value = {}
                result = process_optimize(args, logger)
                assert 'Found 1 knowledgebase configuration(s)' in result


class TestIndexCreation:
    """Test index creation functionality."""
    
    @patch('utils.optimization_manager.KnowledgeBase')
    @patch('utils.optimization_manager.create_missing_indexes')
    def test_optimize_creates_indexes(self, mock_create_indexes, mock_kb):
        """Test that optimize_config creates missing indexes."""
        from utils.optimization_manager import optimize_config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'test.cfg')
            db_path = os.path.join(tmpdir, 'test.db')
            
            # Create config file
            with open(config_path, 'w') as f:
                f.write('[DEFAULT]\nvector_model=test\n')
            
            # Create empty db file
            with open(db_path, 'w') as f:
                f.write('')
            
            # Mock KB instance
            mock_kb_instance = Mock()
            mock_kb_instance.knowledge_base_db = db_path
            mock_kb.return_value = mock_kb_instance
            
            # Mock index creation
            mock_create_indexes.return_value = ['idx_test']
            
            # Run optimization
            changes = optimize_config(config_path, dry_run=False, check_indexes=True)
            
            # Verify index creation was called
            mock_create_indexes.assert_called_once_with(db_path, dry_run=False)


#fin