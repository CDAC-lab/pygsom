import numpy as np
import pandas as pd
import gsom

data_filename = "data/zoo.txt".replace('\\', '/')


if __name__ == '__main__':
    np.random.seed(1)
    df = pd.read_csv(data_filename)
    print(df.shape)
    data_training = df.iloc[:, 1:17]
    gsom_map = gsom.GSOM(.83, 16, max_radius=4)
    gsom_map.fit(data_training.to_numpy(), 100, 50)
    df = df.drop(columns=["label"])
    map_points = gsom_map.predict(df,"Name")
    gsom.plot(map_points, "Name", gsom_map=gsom_map)
    map_points.to_csv("gsom.csv", index=False)



    print("complete")
