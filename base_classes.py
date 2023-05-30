from pathlib import Path


class Dataset:
    """Baseclass for dataset. All datasets should inherit from that."""

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

    def get_sample(self):
        """
        Get one sample from the benchmark data with the ground truth
        :return: sample from the benchmark
        """
        raise NotImplementedError


class RegistrationMethod:
    """
    Base class for registration methods. This should be inherited by all registration approaches.
    """
    name = None

    def __init__(self, parameters):
        """
        Initialize the method with the necessary parameters
        :param parameters: dict of form {own_method_name: {param_name_1: param_1, param_name_2: ...}}
        """
        self.parameters = parameters[self.name]

    def __call__(self, source, target, correspondences, variables=None, inlier_inds=None):
        """
        Computes the registration from source and target given a set of correspondences for the
        given variables.
        :param source: np.array of shape [num_source_points, 3]
        :param target: np.array of shape [num_target_points, 3]
        :param correspondences: np.array of shape [num_correspondences, 2]
        :param variables: list(str()), which can contain 's', 'R', 't'
        :return: instance of RegistrationResult containing the estimated parameter
        """

        raise NotImplementedError
