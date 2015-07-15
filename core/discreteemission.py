from emission import Emission


class DiscreteEmission(Emission):

    def __init__(self, name):
        Emission.__init__(self, name)
        self.type = "DiscreteState"
