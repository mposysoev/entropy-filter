from matplotlib import pyplot as plt
import numpy as np
import sys

from EntropyFilter import EntropyFilter


def main():
    # filename = sys.argv[1]
    filename = "short_data_sample.txt"
    data = np.loadtxt(filename)
    entropy_filter = EntropyFilter.EntropyFilter(window_size=150)
    cleaned_data = entropy_filter.calculate_cleaned_data(data)
    entropy = entropy_filter.calculate_only_entropy(data)

    minimum = np.min(data[:, 1])  # for better visibility
    plt.figure(figsize=[16, 4.5])
    plt.scatter(data[:, 0], data[:, 1] - minimum, s=5, c="royalblue", label="before")
    plt.scatter(data[:, 0], cleaned_data[:] - 2 - minimum, s=3, c="red", label="after")
    plt.scatter(data[:, 0], entropy[:] * 60, s=1, c="green", label="entropy")
    plt.legend()
    # plt.savefig("result.png")
    plt.show()


if __name__ == "__main__":
    main()
