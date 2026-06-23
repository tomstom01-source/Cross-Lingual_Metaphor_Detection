import sys
import os

# Make 'src' importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
