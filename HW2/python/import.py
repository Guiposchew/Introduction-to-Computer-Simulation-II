import pandas as pd
import requests
from io import StringIO
import os

# Create data_prof directory if it doesn't exist
os.makedirs('data_prof', exist_ok=True)

base_url = "https://www.physik.uni-leipzig.de/~janke/teaching/"

sizes = {
    8: "2d_is_c008",
    16: "2d_is_c016",
    32: "2d_is_c032",
    64: "2d_is_c064"
}

for L, filename in sizes.items():
    url = base_url + filename
    print(f"Downloading {url}...")
    
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        print(f"Downloaded {len(r.text)} characters")
        
        # Read as whitespace-separated data
        df = pd.read_csv(StringIO(r.text), sep='\s+', header=None, engine='python')
        print(f"Parsed {len(df)} rows")
        
        # Name columns
        df.columns = ["beta", "Cv"]
        
        # Save CSV in data_prof folder
        outname = f"data_prof/specific_heat_L{L}.csv"
        df.to_csv(outname, index=False)
        
        print(f"Saved {outname}")
    except Exception as e:
        print(f"Error processing L={L}: {e}")