class ParamsSync:

    def __init__(self, step_size: int, sigma: float):

        self.step_size = step_size
        self.sigma = sigma

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
