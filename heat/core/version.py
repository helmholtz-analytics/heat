"""This module contains HeAT's version information."""

major = 1
"""Indicates HeAT's main version."""
minor = 0
"""Indicates feature extension."""
micro = 0
"""Indicates revisions for bugfixes."""
extension = None
"""Indicates special builds, e.g. for specific hardware."""

if not extension:
    __version__ = "{}.{}.{}".format(major, minor, micro)
    """The combined version string, consisting out of major, minor, micro and possibly extension."""
else:
    __version__ = "{}.{}.{}-{}".format(major, minor, micro, extension)
