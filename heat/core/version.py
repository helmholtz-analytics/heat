"""This module contains Heat's version information."""

major: int = 1
"""Indicates Heat's main version."""
minor: int = 2
"""Indicates feature extension."""
micro: int = 2
"""Indicates revisions for bugfixes."""
extension: str = "dev"
"""Indicates special builds, e.g. for specific hardware."""

if not extension:
    __version__: str = "{}.{}.{}".format(major, minor, micro)
    """The combined version string, consisting out of major, minor, micro and possibly extension."""
else:
    __version__: str = "{}.{}.{}-{}".format(major, minor, micro, extension)
