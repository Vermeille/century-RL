from boardrl import models


def test_python_model_presets_construct_models():
    assert models.toy().spec()["dim"] == 16
    assert models.cnn().spec()["backbone"] == "cnn"
    assert models.minimal_lstm().spec()["backbone"] == "lstm"


def test_presets_accept_python_overrides():
    assert models.cnn(dim=32).spec()["dim"] == 32


def test_architectures_dispatch_by_registered_name():
    assert models.make("cnn").spec() == models.cnn().spec()
