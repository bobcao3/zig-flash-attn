import functools
from types import SimpleNamespace

from cuda.core import _tensor_map as tensor_map_types
from cuda.core.utils import StridedMemoryView

BFLOAT16_BYTES = 2
SWIZZLES = {
    0: tensor_map_types.TensorMapSwizzle.NONE,
    32: tensor_map_types.TensorMapSwizzle.SWIZZLE_32B,
    64: tensor_map_types.TensorMapSwizzle.SWIZZLE_64B,
    128: tensor_map_types.TensorMapSwizzle.SWIZZLE_128B,
}
CACHE_LIMIT = 512


ELEMENT_TYPES = {
    2: ("<u2", tensor_map_types.TensorMapDataType.BFLOAT16),
    4: ("<u4", tensor_map_types.TensorMapDataType.FLOAT32),
}


@functools.lru_cache(maxsize=CACHE_LIMIT)
def tiled_tensor_map(pointer, shape, strides, box, swizzle_bytes, element_bytes=BFLOAT16_BYTES):
    array = SimpleNamespace(
        __cuda_array_interface__={
            "version": 3,
            "shape": tuple(shape),
            "strides": tuple(stride * element_bytes for stride in strides),
            "typestr": ELEMENT_TYPES[element_bytes][0],
            "data": (pointer, False),
        }
    )
    view = StridedMemoryView.from_cuda_array_interface(array, stream_ptr=-1)
    return view.as_tensor_map(
        box_dim=tuple(box),
        data_type=ELEMENT_TYPES[element_bytes][1],
        swizzle=SWIZZLES[swizzle_bytes],
        l2_promotion=tensor_map_types.TensorMapL2Promotion.L2_256B,
    )


def tensor_map(pointer, shape, strides, rows, box_columns, swizzle_bytes, element_bytes=BFLOAT16_BYTES, heads=1):
    return tiled_tensor_map(pointer, shape, strides, (1, rows // heads, heads, box_columns), swizzle_bytes, element_bytes)


def as_four_dimensional(view):
    if len(view.shape) == 4:
        return tuple(view.shape), tuple(view.strides)
    rows, heads, columns = view.shape
    return (1, rows, heads, columns), (rows * view.strides[0], *view.strides)
