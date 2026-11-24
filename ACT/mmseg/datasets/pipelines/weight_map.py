import os.path as osp
import mmcv
import numpy as np
from mmseg.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadWeightMap(object):
    """Load weight map from file.
    
    Args:
        data_root (str): The root directory of weight maps.
        suffix (str): The suffix of weight map filenames.
    """

    def __init__(self, data_root, suffix='.png'):
        self.data_root = data_root
        self.suffix = suffix

    def __call__(self, results):
        filename = results['img_info']['filename']
        # filename could be full path in some datasets?
        # Usually it's relative to data_root BUT CustomDataset joins full path.
        # So we should be careful.
        # If LoadImageFromFile loads it, it uses 'filename'.
        base_filename = osp.basename(filename)
        # Remove extension if needed or just replace
        # User script saves as .png
        basename = osp.splitext(base_filename)[0] + self.suffix
        
        filepath = osp.join(self.data_root, basename)
        
        if osp.exists(filepath):
            # Load image using mmcv
            # Note: numpy.fromfile workaround needed for Korean paths?
            # User used it in script. We should try standard first, maybe mmcv handles it?
            # mmcv uses cv2.imread usually.
            # Let's use user's method to be safe.
            try:
                img_array = np.fromfile(filepath, np.uint8)
                img = mmcv.imdecode(img_array, flag='grayscale')
            except:
                img = mmcv.imread(filepath, flag='grayscale')
                
            # Convert to float and un-normalize: 0..255 -> -1..1
            if img is not None:
                weight_map = (img.astype(np.float32) / 255.0) * 2.0 - 1.0
                results['weight_map'] = weight_map
                if 'img_fields' not in results:
                    results['img_fields'] = ['img']
                results['img_fields'].append('weight_map')
            else:
                 # Should we error? Or warn?
                 print(f"Error: Failed to load weight map {filepath}")
                 raise FileNotFoundError(f"Weight map not valid: {filepath}")
            
        else:
            # Fallback if map missing
            print(f"Error: Weight map missing {filepath}")
            raise FileNotFoundError(f"Weight map not found: {filepath}")

        return results

@PIPELINES.register_module()
class PadWeightMap(object):
    """Pad the weight map.
    Should be called AFTER standard Pad.
    """

    def __init__(self, pad_val=-1.0):
        self.pad_val = pad_val

    def __call__(self, results):
        if 'weight_map' not in results:
            return results
        
        # Assume 'pad_shape' is set by previous Pad transform
        if 'pad_shape' in results:
            pad_shape = results['pad_shape'][:2]
            # Check if padding is needed
            if results['weight_map'].shape[:2] != pad_shape:
                 results['weight_map'] = mmcv.impad(
                    results['weight_map'],
                    shape=pad_shape,
                    pad_val=self.pad_val)
        
        return results
