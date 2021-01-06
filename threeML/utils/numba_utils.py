import numba as nb
import numpy as np

_EXPANSION_CONSTANT_ = 1.7


def Vector(numba_type):
    """Generates an instance of a dynamically resized vector numba jitclass."""

    if numba_type in Vector._saved_type:
        return Vector._saved_type[numba_type]

    class _Vector:
        """Dynamically sized arrays in nopython mode."""

        def __init__(self, n):
            """Initialize with space enough to hold n garbage values."""
            self.n = n
            self.m = n
            self.full_arr = np.empty(self.n, dtype=numba_type)

        @property
        def size(self):
            """The number of valid values."""
            return self.n

        @property
        def arr(self):
            """Return the subarray."""
            return self.full_arr[: self.n]

        @property
        def last(self):
            """The last element in the array."""
            if self.n:
                return self.full_arr[self.n - 1]
            else:
                raise IndexError("This numbavec has no elements: cannot return 'last'.")

        @property
        def first(self):
            """The first element in the array."""
            if self.n:
                return self.full_arr[0]
            else:
                raise IndexError(
                    "This numbavec has no elements: cannot return 'first'."
                )

        def clear(self):
            """Remove all elements from the array."""
            self.n = 0
            return self

        def extend(self, other):
            """Add the contents of a numpy array to the end of this Vector.

            Arguments
            ---------
            other : 1d array
                The values to add to the end.
            """
            n_required = self.size + other.size
            self.reserve(n_required)
            self.full_arr[self.size : n_required] = other
            self.n = n_required
            return self

        def append(self, val):
            """Add a value to the end of the Vector, expanding it if necessary."""
            if self.n == self.m:
                self._expand()
            self.full_arr[self.n] = val
            self.n += 1
            return self

        def reserve(self, n):
            """Reserve a n elements in the underlying array.

            Arguments
            ---------
            n : int
                The number of elements to reserve

            Reserving n elements ensures no resize overhead when appending up
            to size n-1 .
            """
            if n > self.m:  # Only change size if we are
                temp = np.empty(int(n), dtype=numba_type)
                temp[: self.n] = self.arr
                self.full_arr = temp
                self.m = n
            return self

        def consolidate(self):
            """Remove unused memory from the array."""
            if self.n < self.m:
                self.full_arr = self.arr.copy()
                self.m = self.n
            return self

        def __array__(self):
            """Array inteface for Numpy compatibility."""
            return self.full_arr[: self.n]

        def _expand(self):
            """Internal function that handles the resizing of the array."""
            self.m = int(self.m * _EXPANSION_CONSTANT_) + 1
            temp = np.empty(self.m, dtype=numba_type)
            temp[: self.n] = self.full_arr[: self.n]
            self.full_arr = temp

        def set_to(self, arr):
            """Make this vector point to another array of values.

            Arguments
            ---------
            arr : 1d array
                Array to set this vector to. After this operation, self.arr
                will be equal to arr. The dtype of this array must be the 
                same dtype as used to create the vector. Cannot be a readonly
                vector.
            """
            self.full_arr = arr
            self.n = self.m = arr.size

        def set_to_copy(self, arr):
            """Set this vector to an array, copying the underlying input.

            Arguments
            ---------
            arr : 1d array
                Array to set this vector to. After this operation, self.arr
                will be equal to arr. The dtype of this array must be the 
                same dtype as used to create the vector.
            """
            self.full_arr = arr.copy()
            self.n = self.m = arr.size

    if numba_type not in Vector._saved_type:
        spec = [("n", nb.uint64), ("m", nb.uint64), ("full_arr", numba_type[:])]
        Vector._saved_type[numba_type] = nb.experimental.jitclass(spec)(_Vector)

    return Vector._saved_type[numba_type]


Vector._saved_type = dict()

VectorUint8 = Vector(nb.uint8)
VectorUint16 = Vector(nb.uint16)
VectorUint32 = Vector(nb.uint32)
VectorUint64 = Vector(nb.uint64)

VectorInt8 = Vector(nb.int8)
VectorInt16 = Vector(nb.int16)
VectorInt32 = Vector(nb.int32)
VectorInt64 = Vector(nb.int64)

VectorFloat32 = Vector(nb.float32)
VectorFloat64 = Vector(nb.float64)

VectorComplex64 = Vector(nb.complex64)
VectorComplex128 = Vector(nb.complex128)

__all_types = tuple(v for k, v in Vector._saved_type.items())


def _isinstance(obj):
    return isinstance(obj, __all_types)


@nb.njit(fastmath=True)
def nb_sum(x):
    return np.sum(x)
