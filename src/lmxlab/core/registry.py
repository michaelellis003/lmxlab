"""Typed component registries for attention, FFN, norm, and position."""

from collections.abc import Callable
from typing import overload


class Registry[T]:
    """A typed registry mapping string names to factory functions.

    Args:
        name: Human-readable name for this registry.

    Example:
        >>> reg = Registry[type]('attention')
        >>> @reg.register('mha')
        ... class MHA: ...
        >>> reg.get('mha')
        <class 'MHA'>
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, T] = {}

    @property
    def name(self) -> str:
        """Registry name."""
        return self._name

    @overload
    def register(self, key: str) -> Callable[[T], T]: ...

    @overload
    def register(self, key: str, value: T) -> T: ...

    def register(
        self, key: str, value: T | None = None
    ) -> T | Callable[[T], T]:
        """Register a component under a key.

        Can be used as a decorator factory or called directly:

            @registry.register('name')
            class MyClass: ...

            registry.register('name', MyClass)

        Args:
            key: String name for lookup.
            value: The component to register. If None, returns
                a decorator that registers the decorated object.

        Returns:
            The registered value, or a decorator if value is None.

        Raises:
            ValueError: If key is already registered.
        """
        if value is not None:
            if key in self._entries:
                raise ValueError(
                    f"{self._name} registry already has key {key!r}"
                )
            self._entries[key] = value
            return value

        def decorator(val: T) -> T:
            if key in self._entries:
                raise ValueError(
                    f"{self._name} registry already has key {key!r}"
                )
            self._entries[key] = val
            return val

        return decorator

    def get(self, key: str) -> T:
        """Look up a component by key.

        Args:
            key: Registered name.

        Returns:
            The registered component.

        Raises:
            KeyError: If key is not found.
        """
        if key not in self._entries:
            available = ", ".join(sorted(self._entries))
            raise KeyError(
                f"{self._name} registry has no key {key!r}. "
                f"Available: [{available}]"
            )
        return self._entries[key]

    def keys(self) -> list[str]:
        """List all registered keys."""
        return sorted(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self._entries))
        return f"Registry({self._name!r}, keys=[{keys}])"
