# mypkg/_lazy.py

import importlib
import importlib.util
from typing import Dict, List, Tuple


def setup_lazy_exports(
    namespace: dict,
    exports: Dict[str, Tuple[str, List[str]]],
):
    def missing_deps(deps: List[str]) -> List[str]:
        return [d for d in deps if importlib.util.find_spec(d) is None]

    available = {name: not missing_deps(deps) for name, (_, deps) in exports.items()}

    namespace["__all__"] = sorted([name for name, ok in available.items() if ok])

    def __getattr__(name: str):
        if name not in namespace["__all__"]:
            if name in exports:
                _, deps = exports[name]
                missing = missing_deps(deps)
                if missing:
                    raise AttributeError(
                        f"{name} unavailable; missing deps: " f"{', '.join(missing)}"
                    )
            raise AttributeError(name)

        mod_path, _ = exports[name]
        mod = importlib.import_module(mod_path, namespace["__name__"])
        obj = getattr(mod, name)

        namespace[name] = obj
        return obj

    def __dir__():
        return sorted(list(namespace.keys()) + namespace["__all__"])

    namespace["__getattr__"] = __getattr__
    namespace["__dir__"] = __dir__
