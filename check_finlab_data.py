
import finlab
from finlab import data
import pandas as pd

# Read API key
with open('finlabapi.txt', 'r') as f:
    api_key = f.read().strip()

finlab.login(api_key)

# Fetch latest revenue data sample
try:
    rev = data.get('monthly_revenue:當月營收')
    print("Revenue Data Shape:", rev.shape)
    print("Revenue Index:", rev.index.name)
    print("Revenue Columns:", rev.columns[:5])
    print(rev.tail())
except Exception as e:
    print("Error fetching revenue:", e)
