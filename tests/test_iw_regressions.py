import numpy as np
import unittest
import dask.array as da

from sarpyx.processor.core.meta import Handler
from sarpyx.processor.core.subaperture_full_img import CombinedSublooking


def check_ifftshift_is_global_across_uneven_chunks():
    obj = CombinedSublooking.__new__(CombinedSublooking)
    for n in (9, 10):
        a = np.arange(n * 3).reshape(n, 3).astype(np.complex64)
        result = obj._ifftshift_freq_axis(da.from_array(a, chunks=(4, 2))).compute()
        np.testing.assert_array_equal(result, np.fft.ifftshift(a, axes=0))
        old = da.from_array(a, chunks=(4, 2)).map_blocks(
            lambda block: np.fft.ifftshift(block, axes=0), dtype=a.dtype).compute()
        assert not np.array_equal(old, result)


def check_shifted_support(start, end, shift, roll_shift):
    n, cols = 11, 3
    spectrum = np.zeros((n, cols), dtype=np.complex64)
    spectrum[start:end + 1] = np.arange(1, end - start + 2)[:, None]
    results = []
    for dask_backend in (False, True):
        obj = CombinedSublooking.__new__(CombinedSublooking)
        obj.nRows, obj.nCols, obj._chunk_cols = n, cols, 2
        obj.iw_support_start_rolled, obj.iw_support_end_rolled = start, end
        obj.iw_roll_shift = roll_shift
        obj.numberOfLooks = 2
        obj.iw_equal_energy_split = False
        obj.iw_energy_compensation = False
        obj._estimate_iw_dc_shift_from_rolled_support = lambda *args: shift
        obj._apply_iw_post_image_corrections_numpy = lambda x: x
        obj._apply_iw_post_image_corrections_dask = lambda x: x
        original = np.roll(spectrum, -roll_shift, axis=0)
        obj.SpectrumOneDimNormDeWe = da.from_array(original, chunks=(n, 2)) if dask_backend else original.copy()
        (obj._generation_dask_iw if dask_backend else obj._generation_numpy_iw)()
        looks = [x.compute() if dask_backend else x for x in obj.Looks]
        reconstructed = np.fft.fftshift(np.fft.fft(sum(looks), axis=0), axes=0)
        np.testing.assert_allclose(reconstructed, np.roll(spectrum, shift - roll_shift, axis=0), atol=1e-5)
        results.append(looks)
    np.testing.assert_allclose(results[0], results[1], atol=1e-6)


def check_annotation_selection(tmp_path):
    annotation = tmp_path / 'product.SAFE' / 'annotation'
    annotation.mkdir(parents=True)
    for swath in ('iw1', 'iw2', 'iw3'):
        for pol in ('vv', 'vh'):
            (annotation / f's1a-{swath}-slc-{pol}-fixture.xml').write_text(f'<root>{swath}-{pol}</root>')
    h = Handler(str(annotation.parent), swath='IW2', polarization='VV')
    h._extract_metadata_from_xml = lambda root: setattr(h, 'selected', root.text)
    h.chain()
    assert h.selected == 'iw2-vv'
    assert h.annotation_file.endswith('s1a-iw2-slc-vv-fixture.xml')
    for selectors in ({'swath': 'IW9', 'polarization': 'VV'},):
        try:
            Handler(str(annotation.parent), **selectors).chain()
        except RuntimeError as exc:
            assert 'Expected one annotation' in str(exc)
        else:
            raise AssertionError('ambiguous or missing annotation must fail')


class RegressionTests(unittest.TestCase):
    def test_ifftshift(self):
        check_ifftshift_is_global_across_uneven_chunks()

    def test_support(self):
        for case in ((1, 5, -3, 2), (5, 9, 3, -3), (2, 7, 1, 0)):
            with self.subTest(case=case):
                check_shifted_support(*case)

    def test_filename_selectors_reach_metadata_handler(self):
        from unittest.mock import patch
        module = 'sarpyx.processor.core.subaperture_full_img'
        with patch(module + '.ut.get_image_shape_from_file', return_value=(8, 2)), \
             patch(module + '.ut.create_complex_image_from_file', return_value=np.zeros((8, 2))), \
             patch(module + '.Handler') as handler:
            meta = handler.return_value
            meta.annotation_file = 'iw2-vv.xml'
            for name in ('PRF', 'AzimBand', 'ChirpBand', 'RangeBand', 'WeightFunctRangeParams',
                         'WeightFunctAzimParams', 'AzimSpacing', 'RangeSpacing'):
                setattr(meta, name, 2.0)
            sub = CombinedSublooking('product.SAFE', i_image='i_IW2_VV.img',
                                     q_image='q_IW2_VV.img', force_dask=False)
            handler.assert_called_once_with('product.SAFE', swath='IW2', polarization='VV', require_unique=True)
            self.assertEqual(sub.annotation_file, 'iw2-vv.xml')
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                CombinedSublooking('product.SAFE', i_image='i_IW2_VV.img',
                                   q_image='q_IW2_VV.img', force_dask=False, safe_swath='IW1')

    def test_annotations(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            check_annotation_selection(Path(folder))


class PublicSelectionTests(unittest.TestCase):
    def test_legacy_warns_and_strict_rejects_ambiguous(self):
        import warnings
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            annotation = Path(tmp)/'annotation'; annotation.mkdir()
            for swath in ('iw1', 'iw2'):
                (annotation/f's1a-{swath}-slc-vv-test.xml').write_text('<root/>')
            h = Handler(tmp); h._extract_metadata_from_xml = lambda root: None
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always'); h.chain()
            self.assertEqual(len(caught),1)
            self.assertTrue(h.annotation_file.endswith('.xml'))
            selected_pol = Handler(tmp, polarization='VV', require_unique=False)
            selected_pol._extract_metadata_from_xml = lambda root: None
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always'); selected_pol.chain()
            self.assertEqual(len(caught), 1)
            self.assertTrue(selected_pol.annotation_file.endswith('-vv-test.xml'))
            with self.assertRaisesRegex(RuntimeError, 'Expected one annotation'):
                Handler(tmp, require_unique=True).chain()
            with self.assertRaisesRegex(RuntimeError, 'Error processing metadata: Expected one annotation') as exc:
                Handler(tmp, polarization='VV').chain()
            self.assertNotIn('Failed to read annotation', str(exc.exception))

    def test_independent_iq_name_validation(self):
        from sarpyx.processor.core.subaperture_full_img import _resolve_iq_selectors as resolve
        self.assertEqual(resolve('i_IW2_VV.img','opaque.img'), ('IW2','VV'))
        self.assertEqual(resolve('opaque.img','q_IW2_VH.img'), ('IW2','VH'))
        for args in [('i_IW2_VV.img','opaque.img','IW1',None),
                     ('i_IW2_VV.img','q_IW1_VV.img',None,None),
                     ('i_IW2_VV.img','q_IW2_VH.img',None,None),
                     ('q_IW2_VV.img','opaque.img',None,None)]:
            with self.subTest(args=args), self.assertRaises(ValueError): resolve(*args)

    def test_public_pipeline_dual_pol_and_preflight(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from sarpyx.processor.core import subaperture_full_img as module
        xml = ('<root><prf>123</prf><imageAnnotation><processingInformation><swathProcParams>'
               '<azimuthProcessing><totalBandwidth>10</totalBandwidth><windowCoefficient>.75</windowCoefficient></azimuthProcessing>'
               '<rangeProcessing><totalBandwidth>20</totalBandwidth><windowCoefficient>.75</windowCoefficient></rangeProcessing>'
               '</swathProcParams></processingInformation></imageAnnotation></root>')
        with TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'scene.data'; data.mkdir(); safe=root/'product.SAFE'; ann=safe/'annotation'; ann.mkdir(parents=True)
            for swath,pol in [('iw2','vv'),('iw2','vh')]:
                (ann/f's1a-{swath}-slc-{pol}-test.xml').write_text(xml)
                for role in 'iq': (data/f'{role}_{swath.upper()}_{pol.upper()}.img').touch()
            patches = (patch.object(module, 'CombinedSublooking'), patch.object(module, '_ensure_subap_output_space'),
                       patch.object(module, '_estimate_subap_output_bytes',return_value=0))
            with patches[0] as sub, patches[1], patches[2]:
                module.do_subaps(str(root/'scene.dim'),str(safe),safe_polarization='VV',update_dim=False)
                self.assertEqual(sub.call_count,1)
                self.assertEqual(sub.call_args.kwargs['safe_swath'],'IW2')
                self.assertEqual(sub.call_args.kwargs['safe_polarization'],'VV')
                sub.reset_mock()
                module.do_subaps(str(root/'scene.dim'),str(safe),update_dim=False)
                self.assertEqual(sub.call_count,2)
                self.assertEqual({call.kwargs['safe_polarization'] for call in sub.call_args_list},{'VV','VH'})
                sub.reset_mock()
                (ann/'s1a-iw2-slc-vv-test.xml').unlink()
                with self.assertRaisesRegex(RuntimeError,'Expected one annotation'):
                    module.do_subaps(str(root/'scene.dim'),str(safe),update_dim=False)
                sub.assert_not_called()

    def test_polarization_only_band_warns_and_keeps_legacy_first_annotation(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from sarpyx.processor.core import subaperture_full_img as module
        with TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'scene.data'; data.mkdir(); safe=root/'product.SAFE'; ann=safe/'annotation'; ann.mkdir(parents=True)
            for role in 'iq': (data/f'{role}_VV.img').touch()
            for swath in ('iw1','iw2'):
                (ann/f's1a-{swath}-slc-vv-test.xml').write_text('<root/>')
            with patch.object(module,'Handler') as handler, patch.object(module,'CombinedSublooking') as sub, \
                 patch.object(module,'_ensure_subap_output_space') as reserve, \
                 patch.object(module,'_estimate_subap_output_bytes', return_value=0):
                handler.return_value.annotation_file = 'first-annotation.xml'
                for name in ('PRF','AzimBand','ChirpBand','RangeBand','WeightFunctRangeParams',
                             'WeightFunctAzimParams','AzimSpacing','RangeSpacing'):
                    setattr(handler.return_value, name, 1.0)
                module.do_subaps(str(root/'scene.dim'),str(safe),update_dim=False)
                handler.assert_called_once_with(str(safe), swath=None, polarization='VV', require_unique=False)
                self.assertEqual(sub.call_count,1)
                self.assertIsNone(sub.call_args.kwargs['safe_swath'])
                self.assertEqual(sub.call_args.kwargs['safe_polarization'],'VV')
                reserve.assert_called_once()

    def test_snapflow_forwards_selectors(self):
        from unittest.mock import patch
        from sarpyx.snapflow.engine import GPT
        from pathlib import Path
        obj = GPT.__new__(GPT)
        with patch('sarpyx.processor.core.subaperture_full_img.do_subaps') as called:
            result = obj.do_subaps(dim_path='scene.dim', safe_path='product.SAFE', safe_swath='IW2', safe_polarization='VV')
        self.assertEqual(result, Path('scene.dim'))
        self.assertEqual(called.call_args.kwargs['safe_swath'],'IW2')
        self.assertEqual(called.call_args.kwargs['safe_polarization'],'VV')


class SupportReconstructionTests(unittest.TestCase):
    def test_odd_even_multilook_complex_tone_and_flip(self):
        for n, start, end, shift, roll in ((10,1,7,-4,2),(11,2,8,3,-2),(10,0,9,13,4)):
            for dask_backend in (False,True):
                with self.subTest(n=n,shift=shift,dask=dask_backend):
                    obj=CombinedSublooking.__new__(CombinedSublooking)
                    obj.nRows=n; obj.nCols=2; obj._chunk_cols=1
                    obj.iw_support_start_rolled=start; obj.iw_support_end_rolled=end; obj.iw_roll_shift=roll
                    obj.numberOfLooks=3; obj.iw_equal_energy_split=True; obj.iw_energy_compensation=False
                    obj.iw_row_equalization=False; obj.iw_flip_output=True
                    obj._estimate_iw_dc_shift_from_rolled_support=lambda *args: shift
                    spectrum=np.zeros((n,2),np.complex64)
                    for j in range(start,end+1): spectrum[j,:]=np.exp(0.43j*j)*(j+1)
                    original=np.roll(spectrum,-roll,axis=0)
                    obj.SpectrumOneDimNormDeWe=da.from_array(original,chunks=(n,1)) if dask_backend else original.copy()
                    (obj._generation_dask_iw if dask_backend else obj._generation_numpy_iw)()
                    looks=[x.compute() if dask_backend else x for x in obj.Looks]
                    expected=np.flip(np.fft.ifft(np.fft.ifftshift(np.roll(original,shift,axis=0),axes=0),axis=0),axis=0)
                    np.testing.assert_allclose(sum(looks),expected,atol=2e-6)


if __name__ == '__main__':
    unittest.main()
