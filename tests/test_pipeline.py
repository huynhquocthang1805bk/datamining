import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from generate_demo import generate
from tgcn.data import load_bundle,TrafficWindows
from tgcn.models import make_model
from tgcn.metrics import horizon_metrics


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.path=Path(cls.temp.name)
        generate(cls.path,nodes=5,steps=160,seed=1)
        cls.bundle=load_bundle(cls.path,'demo',seq_len=8,horizon=2)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_split_no_label_leakage(self):
        for s in ('train','val','test'):
            ds=TrafficWindows(self.bundle,s,8,2)
            lo,hi=self.bundle.splits[s]
            self.assertGreaterEqual(int(ds.starts[0]),lo)
            self.assertLessEqual(int(ds.starts[-1])+2,hi)
        self.assertTrue(np.allclose(self.bundle.means,
                                   self.bundle.values[:self.bundle.splits['train'][1]].mean(0),atol=1e-5))

    def test_shapes_gradients(self):
        ds=TrafficWindows(self.bundle,'train',8,2)
        x,t,y,_=ds[0]
        for name in ('gru','tgcn','tgcn_plus'):
            model=make_model(name,self.bundle,hidden_dim=8,horizon=2)
            prediction=model(x[None],t[None]);self.assertEqual(tuple(prediction.shape),(1,2,5))
            prediction.square().mean().backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_ablations(self):
        ds=TrafficWindows(self.bundle,'train',8,2)
        x,t,_,_=ds[0]
        model=make_model('tgcn_plus',self.bundle,horizon=2,no_time_features=True,
                         no_adaptive_graph=True,no_residual=True)
        self.assertEqual(tuple(model(x[None],t[None]).shape),(1,2,5))

    def test_new_models_and_switches(self):
        ds=TrafficWindows(self.bundle,'train',8,2)
        x,t,_,_=ds[0]
        for name,kw in (('gcn',{}),('gru',{'time_features':True,'residual':True}),
                        ('tgcn',{'time_features':True,'residual':True}),('tgcn',{'no_graph':True})):
            model=make_model(name,self.bundle,hidden_dim=8,horizon=2,seq_len=8,**kw)
            self.assertEqual(tuple(model(x[None],t[None]).shape),(1,2,5))
        # residual with zero head == persistence
        m=make_model('tgcn',self.bundle,hidden_dim=8,horizon=2,residual=True)
        for prm in m.head.parameters(): torch.nn.init.zeros_(prm)
        out=m(x[None],t[None])
        self.assertTrue(torch.allclose(out[0,0],x[-1,:,0]))

    def test_sparse_equals_dense(self):
        ds=TrafficWindows(self.bundle,'train',8,2)
        x,t,_,_=ds[0]
        m=make_model('tgcn',self.bundle,hidden_dim=8,horizon=2)
        sparse=m(x[None],t[None])
        m.used_adjacency=lambda: m.adj
        self.assertTrue(torch.allclose(sparse,m(x[None],t[None]),atol=1e-5))

    def test_paper_protocol_split(self):
        b=load_bundle(self.path,'demo',seq_len=8,horizon=2,protocol='paper')
        T=len(b.values)
        self.assertEqual(b.splits['val'],b.splits['test'])
        self.assertEqual(b.splits['test'][0],int(T*.8)+8)   # context inside the test part
        self.assertAlmostEqual(float(b.scaled.max()),1.0,places=5)

    def test_metrics_paper_definitions(self):
        y=np.array([[[1.,2.],[3.,4.]]]); p=y+np.array([[[.5,-.5],[0.,1.]]])
        m=horizon_metrics(y,p,5)
        e=(p-y).ravel(); yy=y.ravel()
        self.assertAlmostEqual(m['overall']['Accuracy'],1-np.linalg.norm(e)/np.linalg.norm(yy))
        self.assertAlmostEqual(m['overall']['R2'],1-(e**2).sum()/((yy-yy.mean())**2).sum())
        self.assertAlmostEqual(m['cumulative']['10min']['RMSE'],m['overall']['RMSE'])
        self.assertAlmostEqual(horizon_metrics(np.array([[[0.,2.]]]),np.array([[[5.,2.]]]),5,
                                               mask_zero=True)['overall']['MAE'],0.0)

    def test_metrics(self):
        true=np.ones((2,2,5),dtype=np.float32)*10
        p=true+2
        result=horizon_metrics(true,p,15)
        self.assertAlmostEqual(result['overall']['MAE'],2.0)
        self.assertEqual(len(result['per_horizon']),2)

if __name__=='__main__':unittest.main()
