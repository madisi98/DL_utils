from .core import Network


class FCNN(Network):
    def __init__(self, params):
        super().__init__(params)

        self.architecture = params['architecture']

        self.build_model()
        self.compile()

    def build_model(self):
        self.add_layer('Dense', self.architecture)
        self.add_layer('Dense', self.output_shape)


class CNN1D(Network):
    def __init__(self, params):
        super().__init__(params)

        self.architecture = params['architecture']
        self.archNN = params['archNN']
        self.padding = params['padding']

        self.build_model()
        self.compile()

    def build_model(self):
        self.add_layer('Conv1D', self.architecture)
        self.add_layer('Flatten')
        self.add_layer('Dense', self.archNN)
        self.add_layer('Dense', self.output_shape)


class CNN2D(Network):
    def __init__(self, params):
        super().__init__(params)

        self.architecture = params['architecture']
        self.archNN = params['archNN']
        self.padding = params['padding']

        self.build_model()
        self.compile()

    def build_model(self):
        self.add_layer('Conv2D', self.architecture)
        self.add_layer('Flatten')
        self.add_layer('Dense', self.archNN)
        self.add_layer('Dense', self.output_shape)

