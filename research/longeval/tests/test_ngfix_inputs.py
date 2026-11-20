import struct

import numpy as np
import pytest

from research.longeval.ngfix_inputs import read_matrix, write_matrix


@pytest.mark.parametrize('dtype,values', [('<f4', [[.5, -.25], [1., 0.]]), ('<i4', [[0, 200], [3, 4]])])
def test_binary_layout_and_roundtrip(tmp_path, dtype, values):
    path = tmp_path / 'matrix.bin'
    write_matrix(path, values, dtype)
    expected = struct.pack('<II', 2, 2) + np.array(values, dtype=dtype).tobytes()
    assert path.read_bytes() == expected
    np.testing.assert_array_equal(read_matrix(path, dtype), values)
    with pytest.raises(FileExistsError):
        write_matrix(path, values, dtype)


@pytest.mark.parametrize('values,dtype', [([[np.nan]], '<f4'), ([[-1]], '<i4'),
    ([[1.5]], '<i4'), ([[2**32]], '<i4'), ([], '<f4'), ([[1]], '<f8')])
def test_invalid_values_rejected_before_creating_file(tmp_path, values, dtype):
    path = tmp_path / 'bad.bin'
    with pytest.raises(ValueError):
        write_matrix(path, values, dtype)
    assert not path.exists()


@pytest.mark.parametrize('content', [b'', struct.pack('<II', 1, 2) + b'1234',
                                  struct.pack('<II', 0, 2)])
def test_truncated_or_empty_binary_rejected(tmp_path, content):
    path = tmp_path / 'bad.bin'
    path.write_bytes(content)
    with pytest.raises(ValueError):
        read_matrix(path, '<f4')
