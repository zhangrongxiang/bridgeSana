
import json

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BridgeDataset(Dataset):
    """Dataset for Bridge training with source-target image pairs"""

    def __init__(self, data_dir, resolution=1024):
        self.data_dir = Path(data_dir)
        self.resolution = resolution

        jsonl_path = self.data_dir / "train.jsonl"
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_rel = item['src']
        tar_rel = item['tar']

        # Handle path prefix
        data_dir_name = self.data_dir.name
        if src_rel.startswith(data_dir_name + '/'):
            src_rel = src_rel[len(data_dir_name) + 1:]
        if tar_rel.startswith(data_dir_name + '/'):
            tar_rel = tar_rel[len(data_dir_name) + 1:]

        src_path = self.data_dir / src_rel
        tar_path = self.data_dir / tar_rel
        prompt = item['prompt']

        src_image = Image.open(src_path).convert('RGB')
        tar_image = Image.open(tar_path).convert('RGB')

        src_tensor = self.transform(src_image)
        tar_tensor = self.transform(tar_image)

        return {
            'source_images': src_tensor,
            'target_images': tar_tensor,
            'prompts': "Convert the style to 3D Chibi Style",
        }


def collate_fn(examples):
    source_images = torch.stack([example['source_images'] for example in examples])
    target_images = torch.stack([example['target_images'] for example in examples])
    prompts = [example['prompts'] for example in examples]
    return {
        'source_images': source_images,
        'target_images': target_images,
        'prompts': prompts,
    }
