from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

from boardrl.serve import serve
from boardrl.serve.serve import Strategies


def test_populate_strategies_finds_model(tmp_path):
    temp_model = Path('example_model.pth')
    temp_model.touch()
    try:
        s = Strategies()
    finally:
        temp_model.unlink()
    assert any(
        f"model=./{temp_model.name}" in name or f"model={temp_model}" in name
        for name in s.strategies
    )
    assert 'random' in s.strategies


def test_get_strategy_caching_and_invalid():
    s = Strategies()
    strat1 = s.get_strategy('random')
    strat2 = s.get_strategy('random')
    assert strat1 is strat2
    with pytest.raises(HTTPException):
        s.get_strategy('unknown')


def test_get_strategies_endpoint():
    client = TestClient(serve.app)
    response = client.get('/strategies')
    assert response.status_code == 200
    assert response.json() == serve.strategies.strategies


def test_play_one_and_reset_endpoint():
    client = TestClient(serve.app)
    client.get('/reset')
    initial_game = serve.game
    resp = client.post('/play-one', json={'strategy': 'random'})
    assert resp.status_code == 200
    assert isinstance(resp.json()['continue'], bool)
    resp = client.get('/reset')
    assert resp.status_code == 200
    assert resp.json() is True
    assert serve.game is not initial_game


def test_analyze_endpoint():
    client = TestClient(serve.app)
    client.get('/reset')
    resp = client.get('/analyze', params={'strategy': 'random'})
    data = resp.json()
    assert 'moves' in data
    assert isinstance(data['moves'], dict)
