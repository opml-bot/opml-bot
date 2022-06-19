import pandas as pd
from typing import Optional


def prepare_data(path: str,
                 train_size: Optional[float] = 0.8):


    data = pd.read_csv(path)
    target = data["Y"].values
    val = data.drop(["Y"], axis=1).values
    sep = int(val.shape[0]*train_size)
    train_x = val[:sep]
    test_x = val[sep:]
    train_y = target[:sep]
    test_y = target[sep:]

    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    pass