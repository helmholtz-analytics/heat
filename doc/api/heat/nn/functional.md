Module heat.nn.functional
=========================
File containing the heat.nn.functional submodule

Functions
---------

`func_getattr(name)`
:   When a function is called for the heat.nn.functional module it will attempt to run the
    heat.nn.functional module with that name, then, if there is no such heat nn module,
    it will attempt to get the torch.nn.functional module of that name.
