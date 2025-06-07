#!/usr/bin/env python
"""
Unit tests for security_utils module.
Tests path validation with multiple allowed directories and expanded paths.
"""

import os
import pytest
import tempfile
from pathlib import Path

from utils.security_utils import validate_file_path


class TestValidateFilePath:
  """Test cases for validate_file_path function with multiple allowed directories."""
  
  def test_validate_file_path_with_allowed_dirs(self):
    """Test that absolute paths within allowed directories are accepted."""
    with tempfile.TemporaryDirectory() as tmpdir:
      allowed_dirs = [tmpdir, '/var/lib/vectordbs']
      
      # Test path within allowed directory
      test_file = os.path.join(tmpdir, 'test.cfg')
      result = validate_file_path(test_file, ['.cfg'], allowed_dirs=allowed_dirs)
      assert result == test_file
      
      # Test path in second allowed directory
      vectordb_file = '/var/lib/vectordbs/myproject.cfg'
      result = validate_file_path(vectordb_file, ['.cfg'], allowed_dirs=allowed_dirs)
      assert result == vectordb_file
  
  def test_validate_file_path_outside_allowed_dirs(self):
    """Test that absolute paths outside allowed directories are rejected."""
    allowed_dirs = ['/var/lib/vectordbs', '/home/user']
    
    with pytest.raises(ValueError, match="File path outside allowed directories"):
      validate_file_path('/etc/passwd', None, allowed_dirs=allowed_dirs)
  
  def test_validate_file_path_relative_with_allowed_dirs(self):
    """Test that relative paths work with allowed directories."""
    allowed_dirs = ['/var/lib/vectordbs', os.getcwd()]
    
    # Relative path should work
    result = validate_file_path('config.cfg', ['.cfg'], allowed_dirs=allowed_dirs)
    assert result == 'config.cfg'
  
  def test_validate_file_path_empty_allowed_dirs(self):
    """Test behavior with empty allowed_dirs list."""
    with pytest.raises(ValueError, match="Absolute paths not allowed"):
      validate_file_path('/absolute/path.cfg', ['.cfg'], allowed_dirs=[])
  
  def test_validate_file_path_none_in_allowed_dirs(self):
    """Test that None values in allowed_dirs are ignored."""
    allowed_dirs = [None, '/var/lib/vectordbs', None]
    
    result = validate_file_path('/var/lib/vectordbs/test.cfg', ['.cfg'], allowed_dirs=allowed_dirs)
    assert result == '/var/lib/vectordbs/test.cfg'
  
  def test_validate_file_path_backwards_compatibility(self):
    """Test that base_dir still works for backwards compatibility."""
    with tempfile.TemporaryDirectory() as tmpdir:
      test_file = os.path.join(tmpdir, 'test.cfg')
      
      # Using base_dir should still work
      result = validate_file_path(test_file, ['.cfg'], base_dir=tmpdir)
      assert result == test_file
      
      # Path outside base_dir should fail
      with pytest.raises(ValueError, match="File path outside allowed directory"):
        validate_file_path('/etc/passwd', None, base_dir=tmpdir)
  
  def test_validate_file_path_precedence(self):
    """Test that allowed_dirs takes precedence over base_dir."""
    with tempfile.TemporaryDirectory() as tmpdir1:
      with tempfile.TemporaryDirectory() as tmpdir2:
        # allowed_dirs should take precedence
        test_file = os.path.join(tmpdir2, 'test.cfg')
        result = validate_file_path(test_file, ['.cfg'], 
                                   base_dir=tmpdir1,
                                   allowed_dirs=[tmpdir2])
        assert result == test_file
        
        # File in base_dir but not in allowed_dirs should fail
        test_file2 = os.path.join(tmpdir1, 'test.cfg')
        with pytest.raises(ValueError, match="File path outside allowed directories"):
          validate_file_path(test_file2, ['.cfg'], 
                           base_dir=tmpdir1,
                           allowed_dirs=[tmpdir2])
  
  def test_validate_file_path_traversal_with_allowed_dirs(self):
    """Test that path traversal is still blocked with allowed_dirs."""
    allowed_dirs = ['/var/lib/vectordbs']
    
    # Path traversal should still be blocked
    with pytest.raises(ValueError, match="path traversal detected"):
      validate_file_path('/var/lib/vectordbs/../../../etc/passwd', None, allowed_dirs=allowed_dirs)
    
    # Even if the resolved path would be in allowed dirs
    with pytest.raises(ValueError, match="path traversal detected"):
      validate_file_path('/var/lib/../lib/vectordbs/test.cfg', None, allowed_dirs=allowed_dirs)
  
  def test_validate_file_path_symlinks_with_allowed_dirs(self):
    """Test handling of symlinks with allowed directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
      with tempfile.TemporaryDirectory() as outside_dir:
        allowed_dirs = [tmpdir]
        
        # Create a file outside allowed directory
        outside_file = os.path.join(outside_dir, 'outside.cfg')
        Path(outside_file).touch()
        
        # Create symlink inside allowed directory pointing outside
        symlink = os.path.join(tmpdir, 'link.cfg')
        os.symlink(outside_file, symlink)
        
        # The symlink path itself is in allowed dir, so it should pass
        # (actual file access would be blocked at a different layer)
        result = validate_file_path(symlink, ['.cfg'], allowed_dirs=allowed_dirs)
        assert result == symlink


class TestValidateFilePathNewFeatures:
  """Test cases for new allow_absolute and allow_relative_traversal parameters."""
  
  def test_allow_absolute_parameter(self):
    """Test allow_absolute parameter functionality."""
    # Default behavior - absolute paths should be rejected
    with pytest.raises(ValueError, match="Absolute paths not allowed"):
      validate_file_path('/absolute/path.cfg', ['.cfg'])
    
    # With allow_absolute=False (explicit) - should still reject
    with pytest.raises(ValueError, match="Absolute paths not allowed"):
      validate_file_path('/absolute/path.cfg', ['.cfg'], allow_absolute=False)
    
    # With allow_absolute=True - should accept
    result = validate_file_path('/absolute/path.cfg', ['.cfg'], allow_absolute=True)
    assert result == '/absolute/path.cfg'
  
  def test_allow_relative_traversal_parameter(self):
    """Test allow_relative_traversal parameter functionality."""
    # Default behavior - relative traversal should be rejected
    with pytest.raises(ValueError, match="path traversal detected"):
      validate_file_path('../sibling/config.cfg', ['.cfg'])
    
    # With allow_relative_traversal=False (explicit) - should still reject
    with pytest.raises(ValueError, match="path traversal detected"):
      validate_file_path('../sibling/config.cfg', ['.cfg'], allow_relative_traversal=False)
    
    # With allow_relative_traversal=True - should accept
    result = validate_file_path('../sibling/config.cfg', ['.cfg'], allow_relative_traversal=True)
    assert result == '../sibling/config.cfg'
  
  def test_combined_allow_parameters(self):
    """Test combination of allow_absolute and allow_relative_traversal."""
    # Both absolute and relative traversal allowed
    result1 = validate_file_path('/absolute/../path/config.cfg', ['.cfg'], 
                                allow_absolute=True, allow_relative_traversal=True)
    assert result1 == '/absolute/../path/config.cfg'
    
    # Only absolute allowed
    result2 = validate_file_path('/absolute/path/config.cfg', ['.cfg'], 
                                allow_absolute=True, allow_relative_traversal=False)
    assert result2 == '/absolute/path/config.cfg'
    
    # Only relative traversal allowed
    result3 = validate_file_path('../relative/config.cfg', ['.cfg'], 
                                allow_absolute=False, allow_relative_traversal=True)
    assert result3 == '../relative/config.cfg'
  
  def test_relative_traversal_edge_cases(self):
    """Test edge cases for relative traversal detection."""
    # Should allow when enabled
    with pytest.raises(ValueError, match="path traversal detected"):
      validate_file_path('../../deep/traversal.cfg', ['.cfg'])
    
    result = validate_file_path('../../deep/traversal.cfg', ['.cfg'], allow_relative_traversal=True)
    assert result == '../../deep/traversal.cfg'
    
    # Should still allow normal relative paths
    result = validate_file_path('normal/relative.cfg', ['.cfg'])
    assert result == 'normal/relative.cfg'
    
    # Should still allow files with .. in filename (not traversal)
    result = validate_file_path('file..with..dots.cfg', ['.cfg'])
    assert result == 'file..with..dots.cfg'
  
  def test_kb_config_use_case(self):
    """Test the specific use case for KB config files."""
    # Case 1: Absolute path
    result1 = validate_file_path('/var/lib/vectordbs/project/project.cfg', ['.cfg'], 
                                allow_absolute=True, allow_relative_traversal=True)
    assert result1 == '/var/lib/vectordbs/project/project.cfg'
    
    # Case 2: Relative traversal (sibling directories)
    result2 = validate_file_path('../okusimail/okusimail.cfg', ['.cfg'], 
                                allow_absolute=True, allow_relative_traversal=True)
    assert result2 == '../okusimail/okusimail.cfg'
    
    # Case 3: Mixed absolute with traversal
    result3 = validate_file_path('/home/user/../user/projects/kb.cfg', ['.cfg'], 
                                allow_absolute=True, allow_relative_traversal=True)
    assert result3 == '/home/user/../user/projects/kb.cfg'
  
  def test_security_still_enforced(self):
    """Test that other security checks are still enforced with new parameters."""
    # Dangerous characters should still be blocked
    with pytest.raises(ValueError, match="dangerous characters"):
      validate_file_path('../config|evil.cfg', ['.cfg'], allow_relative_traversal=True)
    
    # Wrong extensions should still be blocked
    with pytest.raises(ValueError, match="Invalid file extension"):
      validate_file_path('../config.txt', ['.cfg'], allow_relative_traversal=True)
    
    # Empty paths should still be blocked
    with pytest.raises(ValueError, match="cannot be empty"):
      validate_file_path('', ['.cfg'], allow_absolute=True)

#fin