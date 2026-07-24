from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup  # type: ignore[import-untyped]
from setuptools.command.build_ext import build_ext  # type: ignore[import-untyped]


extensions = [
    Extension("boardrl.cyutils", ["boardrl/cyutils.pyx"]),
    Extension(
        "boardrl.games.century.engine",
        ["boardrl/games/century/engine.pyx"],
        language="c++",
    ),
]


class CleanInplaceBuildExt(build_ext):
    def run(self):
        super().run()
        if not self.inplace:
            return
        for extension in self.extensions:
            Path(self.get_ext_fullpath(extension.name)).unlink(missing_ok=True)


setup(
    cmdclass={"build_ext": CleanInplaceBuildExt},
    ext_modules=cythonize(
        extensions,
        build_dir="build/cythonized",
        compiler_directives={"language_level": "3"},
    ),
)
