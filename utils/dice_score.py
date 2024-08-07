from torch import Tensor


def dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    # sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    # Average of Dice coefficient for all classes
    return dice_coeff(
        input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon
    )


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


######


def tversky_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    alpha: float = 0.5,
    beta: float = 0.5,
    epsilon: float = 1e-6,
):
    # Average of Tversky coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    TP = (input * target).sum(dim=sum_dim)
    FP = ((1 - target) * input).sum(dim=sum_dim)
    FN = ((1 - input) * target).sum(dim=sum_dim)

    tversky = (TP + epsilon) / (TP + alpha * FP + beta * FN + epsilon)
    return tversky.mean()


def multiclass_tversky_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    alpha: float = 0.5,
    beta: float = 0.5,
    epsilon: float = 1e-6,
):
    # Average of Tversky coefficient for all classes
    return tversky_coeff(
        input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon
    )


def tversky_loss(
    input: Tensor,
    target: Tensor,
    multiclass: bool = False,
    alpha: float = 0.5,
    beta: float = 0.5,
):
    # Tversky loss (objective to minimize) between 0 and 1
    fn = multiclass_tversky_coeff if multiclass else tversky_coeff
    return 1 - fn(input, target, reduce_batch_first=True, alpha=alpha, beta=beta)
