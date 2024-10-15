df = pd.read_csv(r"D:\CODING\Python\MetnumUTS\data.csv")
df.set_index("Minute", inplace=True)
x = np.array(df.index)
y = np.array(df["Output (kW)"])