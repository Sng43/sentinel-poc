"""Put project_sentinel/ on sys.path so tests can `import src...` / `import backend...`
regardless of where pytest is invoked from."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
