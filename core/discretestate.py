from state import State


class DiscreteState(State):

    def __init__(self, name):
        State.__init__(self, name)
        self.type = "DiscreteState"