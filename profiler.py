import pstats, cProfile

import pyximport

pyximport.install()
import engine

g = engine.Game()

cProfile.runctx("g.gen_good_move(1000)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
