import numpy as np
import fire


def main(m=0.0):
    m = np.abs(m)
    a = np.random.randn() + m
    print(a)


if __name__ == "__main__":
    fire.Fire(main)
