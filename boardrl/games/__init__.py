import pyximport

pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
from boardrl.games.century.strategies import strategy_from_string
from boardrl.games.century.engine import Century
