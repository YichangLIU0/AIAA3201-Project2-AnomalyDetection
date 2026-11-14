import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from methods.part1.ae import CompactAE
from methods.normal_training import validate_training, save_map


class AETest(unittest.TestCase):
    def test_odd_geometry_has_trainable_exact_size_output(self):
        torch.set_num_threads(2)
        model=CompactAE()
        x=torch.rand(2,3,70,98)
        output=model(x)
        self.assertEqual(output.shape,x.shape)
        loss=(output-x).square().mean(); loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.encoder[0].weight.grad.abs().sum()),0.)

    def test_calibration_and_test_cannot_train(self):
        good={'normal_training_pool':['train1','train2'],'calibration':['cal'],'test':[{'path':'test'}]}
        validate_training(good,['train2','train1'])
        for names in [['train1','cal'],['train1','train1'],['train1','test']]:
            with self.assertRaises(ValueError): validate_training(good,names)
        bad={**good,'normal_training_pool':['train1','cal']}
        with self.assertRaises(ValueError): validate_training(bad,['train1','cal'])

    def test_npz_is_bit_exact_and_refuses_overwrite(self):
        values=np.random.default_rng(1).random((30,40),dtype=np.float32)
        values[0,0]=np.float32(-0.)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'map.npz'; save_map(path,values)
            with np.load(path) as archive:
                np.testing.assert_array_equal(archive['anomaly_map'].view(np.uint32),values.view(np.uint32))
            with self.assertRaises(FileExistsError): save_map(path,values)


if __name__=='__main__': unittest.main()
