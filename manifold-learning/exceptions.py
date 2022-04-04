class EarlyStoppingException(Exception):
    pass


class NanException(Exception):
    pass


class IntractableLikelihoodError(Exception):
    pass


class DatasetNotAvailableError(Exception):
    pass


class NoMeanException(Exception):
    """Exception to be thrown when a mean function doesn't exist."""

    pass