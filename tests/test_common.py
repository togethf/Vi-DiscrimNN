import numpy as np
from commons.utils import resolve_npz
import os
class TestUtils:
    def test_resolvnpz(self):
        a = np.array([1, 2, 3])
        b = np.array(['a', 'b', 'c'])
        c = np.array(['!', '@', '$'])
        np.savez('temp.npz', a=a, b=b, c=c)
        rnpz = resolve_npz('temp.npz')
        for item in rnpz:
            print(f'name: {item[0]}, value: {item[1]}')
        os.remove('temp.npz')
