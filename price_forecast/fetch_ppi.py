"""Download the 7 monthly price indices from FRED into data/ppi/ppi_monthly.csv.

Run on your own machine (FRED was not reachable from the cloud workspace):
    python price_forecast/fetch_ppi.py            # from 2010-01 to latest
Series ids and descriptions live in config.PPI_SERIES.
"""
import io
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"


def fetch_one(sid, start="2010-01-01", tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(URL.format(sid=sid, start=start), headers={"User-Agent": "Mozilla/5.0"})
            txt = urllib.request.urlopen(req, timeout=30).read().decode()
            df = pd.read_csv(io.StringIO(txt))
            date_col = [c for c in df.columns if c.lower() in ("date", "observation_date")][0]
            s = pd.to_numeric(df[sid], errors="coerce")  # FRED writes '.' for missing
            s.index = pd.to_datetime(df[date_col])
            return s.rename(sid)
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise RuntimeError(f"FRED download failed for {sid}: {e}")
            time.sleep(2)


def main(start="2010-01-01"):
    out = pd.concat([fetch_one(s, start) for s in config.PPI_SERIES], axis=1)
    out.index.name = "date"
    config.PPI_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.PPI_CSV)
    print(f"wrote {config.PPI_CSV}: {len(out)} months, {out.index.min().date()} .. {out.index.max().date()}")
    print(out.tail(3).round(1).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
