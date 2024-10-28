import pandas as pd
#
df = pd.read_hdf("./data.h5", key="df6")
print(df)
