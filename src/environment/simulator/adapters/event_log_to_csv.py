def export_event_log_to_csv(event_log, path: str):
    import pandas as pd
    df = pd.DataFrame(event_log)
    df.to_csv(path, index=False)
