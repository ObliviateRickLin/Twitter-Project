import unittest
import sys
import os
import importlib

class TestEnvironment(unittest.TestCase):
    """Test that all required packages are installed"""
    
    def test_required_packages(self):
        """Test that all required packages can be imported"""
        required_packages = [
            "pyspark",
            "pandas",
            "numpy",
            "matplotlib",
            "seaborn",
            "sklearn",
            "wordcloud",
            "pytz",
            "argparse",
            "json",
            "datetime"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.fail(f"Missing packages: {', '.join(missing_packages)}")
    
    def test_project_structure(self):
        """Test that the project structure is correct"""
        # Check for main Python files
        required_files = [
            "data_utils.py",
            "basic_stats.py",
            "feature_engineering.py",
            "model.py",
            "visualization.py",
            "main.py",
            "README.md"
        ]
        
        # Check for directories
        required_dirs = [
            "analysis",
            "analysis/figures",
            "analysis/stats",
            "analysis/models",
            "tests"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(__file__)), file_path)):
                missing_files.append(file_path)
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), dir_path)):
                missing_dirs.append(dir_path)
        
        if missing_files:
            self.fail(f"Missing files: {', '.join(missing_files)}")
        
        if missing_dirs:
            self.fail(f"Missing directories: {', '.join(missing_dirs)}")

if __name__ == '__main__':
    unittest.main() 