"""Task head registry."""

from soma.registry import Registry

task_registry = Registry("tasks")


def task_family_of(name: str) -> str:
    """Return the ``task_family`` of the head registered as ``name``.

    Config and pipeline validation key on the family, not on the registered name, so a
    user-registered subclass (for example a segmentation head with a custom loss) is
    accepted everywhere the built-in head of that family is. Raises ``ValueError`` with
    the registered names when ``name`` is unknown.
    """
    # Importing the package registers the built-in heads; a user module registers its
    # own before the config is validated.
    import soma.tasks  # noqa: F401

    try:
        head_cls = task_registry.get(name)
    except KeyError as exc:
        raise ValueError(f"Unknown task head {name!r}: {exc.args[0]}") from None
    return str(head_cls.task_family)
