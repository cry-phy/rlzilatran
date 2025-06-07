from src.observer import BaseObserver


def elo_rating(
    ratinga: float, ratingb: float, score: float, scaling: float = 400
) -> tuple[float, float]:
    """Generate new ratings for agent A and B using ELO approach."""
    expected: float = 1 / (1 + 10 ** ((ratinga - ratingb) / scaling))
    ratinga += score - expected
    ratingb -= score - expected
    return ratinga, ratingb


class RANKObserver(BaseObserver):
    """Observer playing two Agents against each other and collecting their performance."""
