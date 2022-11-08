class ParamsSync:

    def __init__(self, step_size: int, sigma: float, nr_cores: int):

        self.step_size = step_size
        self.sigma = sigma
        self.nr_cores = nr_cores

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
