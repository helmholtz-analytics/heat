major = 0
minor = 3
micro = 0
extension = None

if not extension:
    __version__ = "{}.{}.{}".format(major, minor, micro)
else:
    __version__ = "{}.{}.{}-{}".format(major, minor, micro, extension)
