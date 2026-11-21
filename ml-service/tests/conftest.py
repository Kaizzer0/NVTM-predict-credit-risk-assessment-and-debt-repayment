import pytest
from sklearn.model_selection import train_test_split

from training import train


@pytest.fixture(scope="session")
def real_sample():
    try:
        frame, *_ = train.load_home_credit(full_audit=False)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    # Draw the fixture only from the official Training partition.
    training, _ = train.split_training_test(frame)
    sample, _ = train_test_split(
        frame.loc[training.x.index], train_size=250, stratify=training.y, random_state=42,
    )
    return sample
