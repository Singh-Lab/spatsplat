import unittest
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from paper.workflows.standardized_maps import splat_maps,cast_maps,check_replay
from paper.workflows.standardized_registration import METHODS,FAMILY_METHODS,barseq_cases,cast_features


class StandardizedRegistrationTests(unittest.TestCase):
    def test_cast_fitting_body_unchanged_except_recording(self):
        import ast
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        old=ast.parse((root/'paper/registration_baselines.py').read_text())
        new=ast.parse((root/'paper/workflows/standardized_cast.py').read_text())
        old=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='run_cast')
        new=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='run_cast_recorded')
        class RemoveRecording(ast.NodeTransformer):
            def visit_Assign(self,node):
                if any(isinstance(t,ast.Name) and t.id=='map_state' for t in node.targets): return None
                return self.generic_visit(node)
            def visit_Expr(self,node):
                call=node.value
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute):
                    base=call.func.value
                    if isinstance(base,ast.Name) and base.id=='map_state': return None
                    if (isinstance(base,ast.Name) and base.id=='torch' and call.func.attr=='save'
                        and call.args and isinstance(call.args[0],ast.Name) and call.args[0].id=='map_state'):
                        return None
                return self.generic_visit(node)
        new=RemoveRecording().visit(new); new.name='run_cast'
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))

    def test_common_roster(self):
        self.assertEqual(len(METHODS),7)
        self.assertEqual(set(METHODS)-{'Unregistered'},set(m for group in FAMILY_METHODS.values() for m in group))
        self.assertNotIn('Splat expression-only rigid',METHODS)

    def test_splat_inverse_affine_and_nonconstant_grid(self):
        a=torch.tensor([[.95,-.1],[.1,.95]])
        t=torch.tensor([.02,-.01])
        residual=torch.tensor([[.02,.01],[-.01,.015]])
        offset=torch.tensor([.005,-.002])
        yy,xx=torch.meshgrid(torch.linspace(0,1,8),torch.linspace(0,1,8),indexing='ij')
        grid=torch.stack([xx,yy],dim=-1)
        d=(grid@residual.T+offset).permute(2,0,1).unsqueeze(0)
        truth=torch.tensor([[.2,.3],[.4,.7],[.7,.5]])
        query=truth@(a+residual).T+t+offset
        state=dict(affine=dict(linear=a,translation=t),warp={
            'base_affine.linear':a,'base_affine.translation':t,'displacement':d})
        maps,audit=splat_maps(query.numpy(),state,dict(nonaffine_inverse_steps=80,nonaffine_inverse_tolerance=1e-6))
        np.testing.assert_allclose(maps['mapped'],truth.numpy(),atol=2e-6)
        np.testing.assert_allclose(maps['affine'],((query-t)@torch.linalg.inv(a).T).numpy(),atol=2e-6)
        self.assertLess(audit['roundtrip_max'],2e-6)

    def test_cast_selected_mesh_composition(self):
        from CAST.CAST_Stack import BSpline_GD_preparation
        extent=torch.tensor([1000.,1000.])
        mesh,_,_,_,_=BSpline_GD_preparation(extent,8,'cpu',None)
        shift=torch.tensor([2.,-3.]).view(2,1,1)
        state=dict(prelocate_theta=torch.tensor([1.,1.,0.,0.,0.]).view(5,1),
            affine_theta=torch.tensor([1.,1.,0.,5.,6.]).view(5,1),
            mean_query=torch.zeros(2,dtype=torch.float64),mean_reference=torch.zeros(2,dtype=torch.float64),
            coordinate_scale=1000.,meshes=[mesh+shift,mesh+shift],offset=torch.zeros(2),max_xy=extent)
        points=np.array([[.2,.3],[.4,.6],[.7,.5]])
        maps=cast_maps(points,state)
        np.testing.assert_allclose(maps['affine'],points+[.005,.006],atol=1e-6)
        np.testing.assert_allclose(maps['mapped'],points+[.009,0.],atol=1e-6)
        state['meshes']=[]; state['affine_theta']=None
        identity=cast_maps(points,state)
        np.testing.assert_allclose(identity['mapped'],points,atol=1e-6)

    def test_native_replay_guard(self):
        native=dict(mapped=np.zeros((3,2)),affine=np.ones((3,2)))
        self.assertEqual(check_replay(native,native)['mapped_replay_max_error'],0)
        with self.assertRaises(ValueError): check_replay(dict(mapped=np.ones((3,2))),native)

    def test_barseq_fit_inputs_exclude_heldout_expression(self):
        rng=np.random.default_rng(7); size=4000
        counts=np.column_stack([np.arange(size)+1,np.ones(size)])
        data=dict(xy=rng.uniform(size=(size,2)),counts=sparse.csr_matrix(counts),genes=np.array(['a','b']),
            obs=pd.DataFrame(dict(section=np.full(size,19),measurement_id=np.arange(size).astype(str))))
        cases=barseq_cases(data,(19,),maximum=size)
        self.assertEqual(len(cases),7)
        for c in cases:
            pair=c['pair']
            train_ids=set(np.concatenate([x.toarray()[:,0] for x in pair['counts']]))
            test_ids=set(np.r_[c['reference_values'][c['reference_eval'],0],c['query_values'][c['query_eval'],0]])
            self.assertFalse(train_ids & test_ids)
            self.assertNotIn('truth',pair); self.assertNotIn('labels',pair)
            self.assertNotIn('shared',pair)
            self.assertEqual(len(set(pair['spots'][0]) & set(pair['spots'][1])),0)

    def test_assay_preprocessing_is_explicit_and_does_not_mutate(self):
        values=np.array([[1.,3.],[2.,2.]])
        pair=dict(counts=[sparse.csr_matrix(values),sparse.csr_matrix(values)])
        np.testing.assert_allclose(cast_features(pair,'Allen')[0],values)
        np.testing.assert_allclose(cast_features(pair,'BARseq')[0],np.log1p(values/4*1e4))
        np.testing.assert_allclose(pair['counts'][0].toarray(),values)


if __name__=='__main__': unittest.main()
