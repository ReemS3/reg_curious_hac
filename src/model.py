from tensorflow.python.keras import Model


class HRL(Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs, training: bool=False):
        ...