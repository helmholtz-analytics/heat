Module heat.core.types
======================
implementations of the different dtypes supported in heat and the

Functions
---------

`can_cast(from_: Union[str, Type[datatype], Any], to: Union[str, Type[datatype], Any], casting: str = 'intuitive') ‑> heat.core.types.bool`
:   Returns True if cast between data types can occur according to the casting rule. If from is a scalar or array
    scalar, also returns True if the scalar value can be cast without overflow or truncation to an integer.

    Parameters
    ----------
    from_ : Union[str, Type[datatype], Any]
        Scalar, data type or type specifier to cast from.
    to : Union[str, Type[datatype], Any]
        Target type to cast to.
    casting: str, optional
        options: {"no", "safe", "same_kind", "unsafe", "intuitive"}, optional
        Controls the way the cast is evaluated
            * "no" the types may not be cast, i.e. they need to be identical
            * "safe" allows only casts that can preserve values with complete precision
            * "same_kind" safe casts are possible and down_casts within the same type family, e.g. int32 -> int8
            * "unsafe" means any conversion can be performed, i.e. this casting is always possible
            * "intuitive" allows all of the casts of safe plus casting from int32 to float32


    Raises
    ------
    TypeError
        If the types are not understood or casting is not a string
    ValueError
        If the casting rule is not understood

    Examples
    --------
    >>> ht.can_cast(ht.int32, ht.int64)
    True
    >>> ht.can_cast(ht.int64, ht.float64)
    True
    >>> ht.can_cast(ht.int16, ht.int8)
    False
    >>> ht.can_cast(1, ht.float64)
    True
    >>> ht.can_cast(2.0e200, "u1")
    False
    >>> ht.can_cast("i8", "i4", "no")
    False
    >>> ht.can_cast("i8", "i4", "safe")
    False
    >>> ht.can_cast("i8", "i4", "same_kind")
    True
    >>> ht.can_cast("i8", "i4", "unsafe")
    True

`canonical_heat_type(a_type: Union[str, Type[datatype], Any]) ‑> Type[heat.core.types.datatype]`
:   Canonicalize the builtin Python type, type string or HeAT type into a canonical HeAT type.

    Parameters
    ----------
    a_type : type, str, datatype
        A description for the type. It may be a a Python builtin type, string or an HeAT type already.
        In the three former cases the according mapped type is looked up, in the latter the type is simply returned.

    Raises
    ------
    TypeError
        If the type cannot be converted.

`heat_type_is_complexfloating(ht_dtype: Type[datatype]) ‑> heat.core.types.bool`
:   Check if Heat type is a complex floating point number, i.e complex64

    Parameters
    ----------
    ht_dtype: ht.dtype
        HeAT type to check

    Returns
    -------
    out: bool
        True if ht_dtype is a complex float, False otherwise

`heat_type_is_exact(ht_dtype: Type[datatype]) ‑> heat.core.types.bool`
:   Check if HeAT type is an exact type, i.e an integer type. True if ht_dtype is an integer, False otherwise

    Parameters
    ----------
    ht_dtype: Type[datatype]
        HeAT type to check

`heat_type_is_inexact(ht_dtype: Type[datatype]) ‑> heat.core.types.bool`
:   Check if HeAT type is an inexact type, i.e floating point type. True if ht_dtype is a float, False otherwise

    Parameters
    ----------
    ht_dtype: Type[datatype]
        HeAT type to check

`heat_type_is_realfloating(ht_dtype: Type[datatype]) ‑> heat.core.types.bool`
:   Check if Heat type is a real floating point number, i.e float32 or float64

    Parameters
    ----------
    ht_dtype: Type[datatype]
        Heat type to check

    Returns
    -------
    out: bool
        True if ht_dtype is a real float, False otherwise

`heat_type_of(obj: Union[str, Type[datatype], Any, Iterable[str, Type[datatype], Any]]) ‑> Type[datatype]`
:   Returns the corresponding HeAT data type of given object, i.e. scalar, array or iterable. Attempts to determine the
    canonical data type based on the following priority list:
        1. dtype property
        2. type(obj)
        3. type(obj[0])

    Parameters
    ----------
    obj : scalar or DNDarray or iterable
        The object for which to infer the type.

    Raises
    ------
    TypeError
        If the object's type cannot be inferred.

`iscomplex(x: dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Test element-wise if input is complex.

    Parameters
    ----------
    x : DNDarray
        The input DNDarray

    Examples
    --------
    >>> ht.iscomplex(ht.array([1 + 1j, 1]))
    DNDarray([ True, False], dtype=ht.bool, device=cpu:0, split=None)

`isreal(x: dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Test element-wise if input is real-valued.

    Parameters
    ----------
    x : DNDarray
        The input DNDarray

    Examples
    --------
    >>> ht.iscomplex(ht.array([1 + 1j, 1]))
    DNDarray([ True, False], dtype=ht.bool, device=cpu:0, split=None)

`issubdtype(arg1: Union[str, Type[datatype], Any], arg2: Union[str, Type[datatype], Any]) ‑> bool`
:   Returns True if first argument is a typecode lower/equal in type hierarchy.

    Parameters
    ----------
    arg1 : type, str, ht.dtype
        A description representing the type. It may be a a Python builtin type, string or an HeAT type already.
    arg2 : type, str, ht.dtype
        A description representing the type. It may be a a Python builtin type, string or an HeAT type already.


    Examples
    --------
    >>> ints = ht.array([1, 2, 3], dtype=ht.int32)
    >>> ht.issubdtype(ints.dtype, ht.integer)
    True
    >>> ht.issubdype(ints.dtype, ht.floating)
    False
    >>> ht.issubdtype(ht.float64, ht.float32)
    False
    >>> ht.issubdtype("i", ht.integer)
    True

`promote_types(type1: Union[str, Type[datatype], Any], type2: Union[str, Type[datatype], Any]) ‑> Type[heat.core.types.datatype]`
:   Returns the data type with the smallest size and smallest scalar kind to which both ``type1`` and ``type2`` may be
    intuitively cast to, where intuitive casting refers to maintaining the same bit length if possible. This function
    is symmetric.

    Parameters
    ----------
    type1 : type or str or datatype
        type of first operand
    type2 : type or str or datatype
        type of second operand

    Examples
    --------
    >>> ht.promote_types(ht.uint8, ht.uint8)
    <class 'heat.core.types.uint8'>
    >>> ht.promote_types(ht.int32, ht.float32)
    <class 'heat.core.types.float32'>
    >>> ht.promote_types(ht.int8, ht.uint8)
    <class 'heat.core.types.int16'>
    >>> ht.promote_types("i8", "f4")
    <class 'heat.core.types.float64'>

`result_type(*arrays_and_types: Tuple[Union[dndarray.DNDarray, Type[datatype], Any]]) ‑> Type[heat.core.types.datatype]`
:   Returns the data type that results from type promotions rules performed in an arithmetic operation.

    Parameters
    ----------
    arrays_and_types: List of arrays and types
        Input arrays, types or numbers of the operation.

    Examples
    --------
    >>> ht.result_type(ht.array([1], dtype=ht.int32), 1)
    ht.int32
    >>> ht.result_type(ht.float32, ht.array(1, dtype=ht.int8))
    ht.float32
    >>> ht.result_type("i8", "f4")
    ht.float64

Classes
-------

`bool(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The boolean datatype in Heat

    ### Ancestors (in MRO)

    * heat.core.types.datatype

    ### Static methods

    `char() ‑> str`
    :   Datatype short-hand name

    `torch_type() ‑> torch.dtype`
    :   Torch Datatype

`bool_(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The boolean datatype in Heat

    ### Ancestors (in MRO)

    * heat.core.types.datatype

`complex128(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The complex 128 bit datatype. Both real and imaginary are 64 bit floating point

    ### Ancestors (in MRO)

    * heat.core.types.complex
    * heat.core.types.number
    * heat.core.types.datatype

    ### Static methods

    `char()`
    :   Datatype short-hand name

    `torch_type()`
    :   Torch Datatype

`cdouble(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The complex 128 bit datatype. Both real and imaginary are 64 bit floating point

    ### Ancestors (in MRO)

    * heat.core.types.complex
    * heat.core.types.number
    * heat.core.types.datatype

`complex64(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The complex 64 bit datatype. Both real and imaginary are 32 bit floating point

    ### Ancestors (in MRO)

    * heat.core.types.complex
    * heat.core.types.number
    * heat.core.types.datatype

    ### Static methods

    `char()`
    :   Datatype short-hand name

    `torch_type()`
    :   Torch Datatype

`cfloat(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The complex 64 bit datatype. Both real and imaginary are 32 bit floating point

    ### Ancestors (in MRO)

    * heat.core.types.complex
    * heat.core.types.number
    * heat.core.types.datatype

    ### Static methods

    `char()`
    :   Datatype short-hand name

    `torch_type()`
    :   Torch Datatype

`csingle(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The complex 64 bit datatype. Both real and imaginary are 32 bit floating point

    ### Ancestors (in MRO)

    * heat.core.types.complex
    * heat.core.types.number
    * heat.core.types.datatype

`datatype(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   Defines the basic heat data types in the hierarchy as shown below. Design inspired by the Python package numpy.
    As part of the type-hierarchy: xx -- is bit-width

        - generic

            - bool, bool_ (kind=?)

            - number

                - integer

                    - signedinteger (intxx)(kind=b, i)

                        - int8, byte

                        - int16, short

                        - int32, int

                        - int64, long
                    - unsignedinteger (uintxx)(kind=B, u)

                        - uint8, ubyte
                - floating (floatxx) (kind=f)

                    - float32, float, float_

                    - float64, double (double)
            - flexible (currently unused, placeholder for characters)

    ### Descendants

    * heat.core.types.bool
    * heat.core.types.flexible
    * heat.core.types.number

    ### Static methods

    `char() ‑> NotImplemented`
    :   Datatype short-hand name

    `torch_type() ‑> NotImplemented`
    :   Torch Datatype

`flexible(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The general flexible datatype. Currently unused, placeholder for characters

    ### Ancestors (in MRO)

    * heat.core.types.datatype

`float32(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The 32 bit floating point datatype

    ### Ancestors (in MRO)

    * heat.core.types.floating
    * heat.core.types.number
    * heat.core.types.datatype

    ### Static methods

    `char() ‑> str`
    :   Datatype short-hand name

    `torch_type() ‑> torch.dtype`
    :   Torch Datatype

`float(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The 32 bit floating point datatype

    ### Ancestors (in MRO)

    * heat.core.types.floating
    * heat.core.types.number
    * heat.core.types.datatype

    ### Static methods

    `char() ‑> str`
    :   Datatype short-hand name

    `torch_type() ‑> torch.dtype`
    :   Torch Datatype

`float_(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The 32 bit floating point datatype

    ### Ancestors (in MRO)

    * heat.core.types.floating
    * heat.core.types.number
    * heat.core.types.datatype

`float64(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The 64 bit floating point datatype

    ### Ancestors (in MRO)

    * heat.core.types.floating
    * heat.core.types.number
    * heat.core.types.datatype

    ### Static methods

    `char() ‑> str`
    :   Datatype short-hand name

    `torch_type() ‑> torch.dtype`
    :   Torch Datatye

`double(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The 64 bit floating point datatype

    ### Ancestors (in MRO)

    * heat.core.types.floating
    * heat.core.types.number
    * heat.core.types.datatype

    ### Static methods

    `torch_type() ‑> torch.dtype`
    :   Torch Datatye

`floating(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The general floating point datatype class.

    ### Ancestors (in MRO)

    * heat.core.types.number
    * heat.core.types.datatype

    ### Descendants

    * heat.core.types.float32
    * heat.core.types.float64

`int16(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   16 bit signed integer datatype

    ### Ancestors (in MRO)

    * heat.core.types.signedinteger
    * heat.core.types.integer
    * heat.core.types.number
    * heat.core.types.datatype

    ### Static methods

    `char() ‑> str`
    :   Datatype short-hand name

    `torch_type() ‑> torch.dtype`
    :   Torch Datatype

`short(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   16 bit signed integer datatype

    ### Ancestors (in MRO)

    * heat.core.types.signedinteger
    * heat.core.types.integer
    * heat.core.types.number
    * heat.core.types.datatype

`int32(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   32 bit signed integer datatype

    ### Ancestors (in MRO)

    * heat.core.types.signedinteger
    * heat.core.types.integer
    * heat.core.types.number
    * heat.core.types.datatype

    ### Static methods

    `char() ‑> str`
    :   Datatype short-hand name

    `torch_type() ‑> torch.dtype`
    :   Torch Datatype

`int(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   32 bit signed integer datatype

    ### Ancestors (in MRO)

    * heat.core.types.signedinteger
    * heat.core.types.integer
    * heat.core.types.number
    * heat.core.types.datatype

`int64(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   64 bit signed integer datatype

    ### Ancestors (in MRO)

    * heat.core.types.signedinteger
    * heat.core.types.integer
    * heat.core.types.number
    * heat.core.types.datatype

    ### Static methods

    `char() ‑> str`
    :   Datatype short-hand name

    `torch_type() ‑> torch.dtype`
    :   Torch Datatype

`long(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   64 bit signed integer datatype

    ### Ancestors (in MRO)

    * heat.core.types.signedinteger
    * heat.core.types.integer
    * heat.core.types.number
    * heat.core.types.datatype

`int8(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   8 bit signed integer datatype

    ### Ancestors (in MRO)

    * heat.core.types.signedinteger
    * heat.core.types.integer
    * heat.core.types.number
    * heat.core.types.datatype

    ### Static methods

    `char() ‑> str`
    :   Datatype short-hand name

    `torch_type() ‑> torch.dtype`
    :   Torch Datatype

`byte(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   8 bit signed integer datatype

    ### Ancestors (in MRO)

    * heat.core.types.signedinteger
    * heat.core.types.integer
    * heat.core.types.number
    * heat.core.types.datatype

`integer(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The general integer datatype. Specific integer classes inherit from this.

    ### Ancestors (in MRO)

    * heat.core.types.number
    * heat.core.types.datatype

    ### Descendants

    * heat.core.types.signedinteger
    * heat.core.types.unsignedinteger

`number(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The general number datatype. Integer and Float classes will inherit from this.

    ### Ancestors (in MRO)

    * heat.core.types.datatype

    ### Descendants

    * heat.core.types.complex
    * heat.core.types.floating
    * heat.core.types.integer

`signedinteger(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The general signed integer datatype.

    ### Ancestors (in MRO)

    * heat.core.types.integer
    * heat.core.types.number
    * heat.core.types.datatype

    ### Descendants

    * heat.core.types.int16
    * heat.core.types.int32
    * heat.core.types.int64
    * heat.core.types.int8

`uint8(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   8 bit unsigned integer datatype

    ### Ancestors (in MRO)

    * heat.core.types.unsignedinteger
    * heat.core.types.integer
    * heat.core.types.number
    * heat.core.types.datatype

    ### Static methods

    `char() ‑> str`
    :   Datatype short-hand name

    `torch_type() ‑> torch.dtype`
    :   Torch Datatype

`ubyte(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   8 bit unsigned integer datatype

    ### Ancestors (in MRO)

    * heat.core.types.unsignedinteger
    * heat.core.types.integer
    * heat.core.types.number
    * heat.core.types.datatype

`unsignedinteger(*value, device: Optional[Union[str, devices.Device]] = None, comm: Optional[communication.Communication] = None)`
:   The general unsigned integer datatype

    ### Ancestors (in MRO)

    * heat.core.types.integer
    * heat.core.types.number
    * heat.core.types.datatype

    ### Descendants

    * heat.core.types.uint8
