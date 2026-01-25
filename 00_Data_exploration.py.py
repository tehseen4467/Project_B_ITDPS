


import pandas as pd

user = pd.read_csv("user_activity_logs.csv")
network = pd.read_csv("network_logs.csv")
hr = pd.read_csv("hr_context_data.csv")

print(user.head())
print( network.head())
print(hr.head())



