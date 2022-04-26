from torch.utils.data import Dataset
from PIL import Image
import sys
sys.path.extend(['.', '..'])
from utils.data_utils import make_dataset


class InferenceDataset(Dataset):

	def __init__(self, root, opts, transform=None):
		self.paths = sorted(make_dataset(root))
		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = Image.open(from_path).convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im
