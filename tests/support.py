import os
import tempfile
from pathlib import Path
WORK=tempfile.TemporaryDirectory()
os.environ['DATABASE_URL']=f'sqlite+aiosqlite:///{WORK.name}/tests.db'
os.environ['MODEL_PATH']=str(Path(WORK.name)/'model.pkl')
