"""femsim: public name of the package. `femsim.X` is `Source.X` (the same module objects, not copies, so enums and classes compare
equal whichever name imported them):

    from femsim.core import Model
    from femsim.enums import ElementType
"""

import importlib
import importlib.abc
import importlib.util
import sys

import Source

__version__ = "0.1.0"
__path__ = []          # a package whose submodules are found by the finder below


class _AliasLoader(importlib.abc.Loader):

    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("femsim."):
            return None
        real = "Source." + fullname[len("femsim."):]
        try:
            real_spec = importlib.util.find_spec(real)
        except ModuleNotFoundError:
            return None
        if real_spec is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(real), is_package=real_spec.submodule_search_locations is not None)


sys.meta_path.insert(0, _AliasFinder())
