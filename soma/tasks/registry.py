"""Task head registry."""

from soma.registry import Registry

#: The task-head registry: the ``heads`` domain of :func:`soma.components`.
task_registry = Registry("heads", bundled_module="soma.benchmarks")


def task_head_class(name: str) -> type:
    """Return the head class ``name`` names: a registry key or a dotted import path.

    Raises ``ValueError`` with the registered names when ``name`` is unknown.
    """
    # Importing the package registers the built-in heads; a user module registers its
    # own before the config is validated.
    import soma.tasks  # noqa: F401

    try:
        return task_registry.resolve(name)
    except KeyError as exc:
        raise ValueError(f"Unknown task head {name!r}: {exc.args[0]}") from None


def task_family_of(name: str) -> str:
    """Return the ``task_family`` of the head registered as ``name``.

    Config and pipeline validation key on the family, not on the registered name, so a
    user-registered subclass (for example a segmentation head with a custom loss) is
    accepted everywhere the built-in head of that family is. Raises ``ValueError`` with
    the registered names when ``name`` is unknown.
    """
    return str(task_head_class(name).task_family)
