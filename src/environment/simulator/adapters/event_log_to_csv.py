from pathlib import Path
import pandas as pd

def export_event_log_to_csv(event_log, path: str):
    path_obj = Path(path)

    # Create directory if it doesn't exist
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(event_log)
    df.to_csv(path_obj, index=False)