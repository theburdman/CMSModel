from __future__ import annotations
from pathlib import Path
import pandas as pd

class LocalStorageClient:
    def __init__(self, base_dir: str | Path = '.'):
        self.base_dir = Path(base_dir)

    def read_csv(self, name: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(self.base_dir / name, **kwargs)

    def write_csv(self, df: pd.DataFrame, name: str, **kwargs) -> None:
        path = self.base_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, **kwargs)
