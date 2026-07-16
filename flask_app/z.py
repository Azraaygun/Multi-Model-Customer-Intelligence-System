import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://localhost/customer_intelligence')

df = pd.read_sql('SELECT * FROM customers LIMIT 10', engine)

print(df)