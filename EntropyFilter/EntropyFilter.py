import numpy as np


class EntropyFilter:
    def __init__(self, window_size) -> None:
        self.window_size: int = window_size

    @staticmethod
    def entropy_log(p: float, w: int) -> float:
        """
        https://en.wikipedia.org/wiki/Entropy_(information_theory)

        p -- probability
        """
        if p == 0.0:
            return 0.0
        else:
            return p / float(w) * np.log(p / float(w))

    @staticmethod
    def prepare_data(data: np.ndarray[np.float64], minimum) -> np.ndarray[np.float64]:
        if minimum > 0.0:
            data[:, 1] = data[:, 1] - minimum
        else:
            data[:, 1] = data[:, 1] + minimum
        return data

    def calculate_cleaned_data(self, data):
        minimum = np.min(data[:, 1])
        data = self.prepare_data(data, minimum)
        entropy = self._calculate_windowed_entropy(data)
        entropy = np.insert(entropy, 0, np.zeros(self.window_size))
        entropy_mask = self._create_entropy_mask(entropy)
        data_cleaned = self._apply_entropy_mask(data[:, 1], entropy_mask)
        data_cleaned = self._fill_zeros(data_cleaned)
        if minimum > 0.0:
            data[:, 1] = data[:, 1] + minimum
        else:
            data[:, 1] = data[:, 1] - minimum
        return data_cleaned[:] + minimum

    def calculate_only_entropy(self, data):
        entropy = self._calculate_windowed_entropy(data)
        entropy = np.insert(entropy, 0, np.zeros(self.window_size))
        return entropy

    def _calculate_windowed_entropy(self, data):
        entropy = np.array([])
        amount_of_states = int(np.max(data))
        min_value = np.min(data)

        for i in range(len(data) - self.window_size):
            window = data[i : i + self.window_size, 1]
            value_counts = np.bincount(window.astype(int), minlength=amount_of_states)
            res = -np.sum(
                [self.entropy_log(p, self.window_size) for p in value_counts if p > 0]
            )
            entropy = np.append(entropy, res)

        return entropy

    @staticmethod
    def _create_entropy_mask(entropy):
        return (entropy != 0).astype(float)

    @staticmethod
    def _apply_entropy_mask(data, entropy_mask):
        return data - data * entropy_mask

    @staticmethod
    def _fill_zeros(data):
        for i in range(len(data)):
            if data[i] == 0.0:
                data[i] = data[i - 1]
        return data
