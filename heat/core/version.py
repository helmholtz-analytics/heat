"""This module contains HeAT's version information."""

major: int = 1
"""Indicates HeAT's main version."""
minor: int = 1
"""Indicates feature extension."""
micro: int = 0
"""Indicates revisions for bugfixes."""
extension: str = None
"""Indicates special builds, e.g. for specific hardware."""

if not extension:
    __version__: str = "{}.{}.{}".format(major, minor, micro)
    """The combined version string, consisting out of major, minor, micro and possibly extension."""
else:
    __version__: str = "{}.{}.{}-{}".format(major, minor, micro, extension)
