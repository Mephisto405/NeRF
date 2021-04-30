import torch


def MSE(im, ref, reduce=True):
    """Mean-squared error between images.
    Args:
        im(np.array): image to test.
        ref(np.array): reference for the comparison.
    Returns:
        (float) error value.
    """
    return torch.mean((im - ref) ** 2)

def L1(im, ref, reduce=True):
    """Absolute error between images.
    Args:
        im(np.array): image to test.
        ref(np.array): reference for the comparison.
    Returns:
        (float) error value.
    """
    return torch.mean(torch.abs(im - ref))