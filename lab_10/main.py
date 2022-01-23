import numpy as np


def sig(x):
    """

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray : sigmoid function for x
    """
    return 1 / (1 + np.exp(-x))


def derivative_sig(x):
    """
    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray : sigmoid derivative for x
    """
    return x * (1 - x)


class Neuron:
    def __init__(self):
        self.w = np.random.uniform(size=2)
        self._b = np.random.uniform(size=1)
        self._ins = []
        self.out = 0

    def forward(self, x):
        """
        output for neuron
        Parameters
        ----------
        x : list
            input value [1, 0] for example

        Returns
        -------
        self.out : numpy.float64
            output of one neuron

        """
        #sig(np.dot(self.weights, x) + self._b)`
        self._ins = x
        self.out = sig(np.dot(self.w, x) + self._b)  # TODO: dot
        return self.out

    def backward(self, d_err, lr):
        """
        back propagation for neuron
        Parameters
        ----------
        d_err : numpy.ndarray
            error from previous layer (passing back)
        lr : float
            learning rate

        Returns
        -------
        error : list
            list of error for next layer (passing back)

        """
        loss = derivative_sig(self.out) * d_err
        error = loss * self.w
        self.w[0] += self._ins[0] * loss * lr
        self.w[1] += self._ins[1] * loss * lr
        self._b += loss * lr
        return error


class Model:
    def __init__(self):
        self._h = [Neuron(), Neuron()]
        self._o = Neuron()
        self.input = []
        self.xor_table = np.array([[0], [1], [1], [0]])
        self._learn_rate = 0.1
        self._local_results = []

    def forward(self, x):
        """

        Parameters
        ----------
        x : list
            input value [1, 0] for example

        Returns
        -------
        res : numpy.ndarray
            res of work with input values
        """
        neuron_outs = []
        for n in self._h:
            neuron_outs.append(n.forward(x))
        res = self._o.forward(neuron_outs)
        return res

    def backward(self, d_err):
        """
        back propagation for network
        Parameters
        ----------
        d_err: numpy.ndarray

        """
        hidden_layer_loss = self._o.backward(d_err, self._learn_rate)
        self._h[0].backward(hidden_layer_loss[0], self._learn_rate)
        self._h[1].backward(hidden_layer_loss[1], self._learn_rate)

    def fit(self, ins, outs, epochs=10000):
        """

        Parameters
        ----------
        ins: list
        outs: list
        epochs: int

        """
        self.input = np.array(ins)
        self.xor_table = np.array(outs)
        for epoch in range(epochs):
            self._local_results = []
            for i in range(4):
                result = self.forward(self.input[i])
                self._local_results.append(result)
                self.backward(self.xor_table[i] - result)

    def predict(self, data):
        """

        Parameters
        ----------
        data: list

        """
        for i in data:
            print(self.forward(i), end='')


model = Model()
model.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]])
model.predict([[0, 0], [0, 1], [1, 0], [1, 1]])
