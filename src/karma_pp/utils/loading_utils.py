# copy pasted since 2007

import sys
import traceback
from typing import Any, Mapping, TypedDict

import structlog

__all__ = [
    "Config",
    "import_name",
    "instantiate",
]

log = structlog.get_logger(__name__)


class Config(TypedDict):
    """Configuration for dynamically instantiated objects."""

    code: str
    parameters: dict[str, Any]


def instantiate(function_name: str, parameters: Mapping[str, Any]) -> Any:
    try:
        function = import_name(function_name)
    except ValueError as e:
        log.warning("instantiate_import_failed", function_name=function_name, error=str(e))
        msg = f"instantiate(): Cannot find the function or constructor {function_name!r}."
        raise ValueError(msg) from e

    try:
        # XXX TypeError is too broad, we should bind the params explicitly
        result = function(**parameters)
        return result
    except TypeError:
        params = ", ".join(["{}={!r}".format(k, v) for (k, v) in parameters.items()])
        msg = "instantiate(): Could not call function {!r}\n with params {}:".format(
            function_name,
            params,
        )
        msg += "\n" + traceback.format_exc()
        log.warning("instantiate_call_failed", function_name=function_name, params=list(parameters.keys()))
        raise ValueError(msg)


def import_name(name: str) -> Any:
    """
    Loads the python object with the given name.

    Note that "name" might be "module.module.name" as well.
    """
    expected = (ImportError,)
    try:
        obj = __import__(name, fromlist=["dummy"])
        return obj
    except expected:
        # split in (module, name) if we can
        if "." in name:
            tokens = name.split(".")
            field = tokens[-1]
            module_name = ".".join(tokens[:-1])

            # other method, don't assume that in "M.x", "M" is a module.
            # It could be a class as well, and "x" be a staticmethod.
            try:
                module = import_name(module_name)
            except ImportError as e:
                msg = "Cannot load {!r} (tried also with {!r}):\n".format(name, module_name)
                msg += "\n" + traceback.format_exc()
                raise ValueError(msg)

            if field not in module.__dict__:
                msg = "No field  %r\n" % (field)
                msg += " found in %r." % (module)
                raise ValueError(msg)

            f = module.__dict__[field]

            # "staticmethod" are not functions but descriptors, we need extra magic
            if isinstance(f, staticmethod):
                return f.__get__(module, None)
            else:
                return f

        else:
            msg = "Cannot import name %r." % (name)
            msg += "\n" + traceback.format_exc()
            raise ValueError(msg + f" path = {sys.path}")
