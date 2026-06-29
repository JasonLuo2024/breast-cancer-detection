"""Quick setup verification script."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from tools.cnn_tools import get_vindr_cnn_info
from tools.dicom_tools import get_vindr_dataset_info

info = get_vindr_cnn_info()
ds = get_vindr_dataset_info()

print("=" * 50)
print("SETUP CHECK")
print("=" * 50)
print(f"GPU          : {info['gpu_name']}")
print(f"Device       : {info['device']}")
print(f"Train PNGs   : {info['train_images']}")
print(f"Val PNGs     : {info['val_images']}")
print(f"DICOM images : {ds.get('total_images', 'N/A')}")
print(f"Cancer rate  : {ds.get('cancer_rate_pct', 'N/A')}%")
print(f"Architectures: {', '.join(info['available_architectures'])}")
print(f"Loss fn      : {info['loss_function']}")
print("=" * 50)
print("READY TO TRAIN")
