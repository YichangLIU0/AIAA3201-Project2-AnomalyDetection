from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os
import numpy as np
from shared_protocol.protocol import sha256
from methods.part1.map_storage import MapStorage


class MapStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.baseline=self.root/'base';self.output=self.root/'output'
        self.baseline.mkdir();self.output.mkdir()
        self.store=MapStorage(self.baseline,self.output)

    def tearDown(self):
        self.temp.cleanup()

    def save(self,name,value):
        path=self.output/name;np.save(path,np.full((16,32),value,dtype=np.float32))
        return path,sha256(path)

    def test_identical_bytes_share_and_different_maps_stay_independent(self):
        a,h=self.save('a.npy',.2);b,hb=self.save('b.npy',.2);c,hc=self.save('c.npy',.3)
        self.store.share(a,h);self.store.share(b,hb);self.store.share(c,hc)
        self.assertTrue(os.path.samefile(a,b));self.assertFalse(os.path.samefile(a,c))
        self.assertEqual(sha256(b),hb);self.assertEqual(self.store.stats()['shared_files'],1)

    def test_changed_canonical_refused_target_preserved(self):
        a,h=self.save('a.npy',.2);b,_=self.save('b.npy',.2)
        self.store.share(a,h);a.write_bytes(b'changed canonical')
        before=b.read_bytes()
        with self.assertRaises(ValueError):self.store.share(b,h)
        self.assertEqual(b.read_bytes(),before)

    def test_failed_link_preserves_target_and_cleans_temp(self):
        a,h=self.save('a.npy',.2);b,_=self.save('b.npy',.2)
        self.store.share(a,h);before=b.read_bytes()
        with patch('methods.part1.map_storage.os.link',side_effect=OSError('simulated failure')):
            with self.assertRaises(OSError):self.store.share(b,h)
        self.assertEqual(b.read_bytes(),before);self.assertFalse(list(self.output.glob('*.tmp')))

    def test_outside_output_refused(self):
        path=self.baseline/'x.npy';np.save(path,np.zeros((16,32),np.float32))
        with self.assertRaises(ValueError):self.store.share(path,sha256(path))


if __name__=='__main__':
    unittest.main()
