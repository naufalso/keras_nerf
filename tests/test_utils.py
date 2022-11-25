import pytest
from keras_nerf.data.utils import *


def test_get_focal_from_fov():
    fov = 0.6911112070083618
    width = 100
    expected_focal = 138.88887889922103

    assert pytest.approx(get_focal_from_fov(fov, width)) == expected_focal
