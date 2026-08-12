import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torchvision
t = torchvision.transforms.ToTensor()
from torchvision.transforms import v2
from torchvision import transforms
from transformers import AutoImageProcessor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import decode_image
import polars as pl

def normalize_scale_for_test(im):
    sizes = {160:160, 238:238, 512:512}
    t = v2.functional.center_crop(im, sizes[im.shape[-2]])
    t = v2.functional.resize(t, (224,224))
    return t

class self_normalize(object):
    """We're finetuning and this works better than imagenet norms for cell data
    """
    def __call__(self, x):
        m = x.mean((-2, -1), keepdim=True)
        s = x.std((-2, -1), unbiased=False, keepdim=True)
        x -= m
        x /= s + 1e-7
        return x

class MultiChannelDataset(Dataset):
    def __init__(self, metadata, im_dir, transform=None) -> None:
        
        self.transform = transform
        self.samples, self.labels, self.class_map = self.generate_samples(metadata)
        self.num_classes = len(self.class_map.keys())
        self.im_dir = im_dir
        
    def read_im(self, file_path: str):
        return decode_image(os.path.join(self.im_dir, file_path))
 
    def generate_samples(self, config_path):
        """Gross function to get 2 channels to train with. It works better perf wise for the models, and is easier to sample in zarr versus the zip version I used here.
        Ignore this code since you likely won't be using a dataset like this T_T.

        Args:
            config_path (str): Path to the config file
        """
        proper_path = os.path.abspath(os.path.expanduser(config_path))
        config = pl.read_csv(proper_path)
        config = config.with_columns(pl.col('storage.path').str.split('/').list.get(1).alias('study'))
        sampled_ids = config.select(["study", "imaging.multi_channel_id"]).unique().sample(fraction=1.0, shuffle=True, seed=42).group_by("study").head(1000)['imaging.multi_channel_id']
        config = config.filter(pl.col('imaging.multi_channel_id').is_in(sampled_ids))
        unique_classes = set(config['label'].to_list())
        class_map = {}
        for idx, unique in enumerate(unique_classes):
            class_map[unique] = idx
            
        samples = []
        labels = []
        for _, group in config.group_by('imaging.multi_channel_id'):
            train_group = group.sample(n=2)
            train_paths = tuple(train_group['storage.path'].to_list())
            samples.append(train_paths)
            labels.append(train_group['label'][0])
        return samples, labels, class_map
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        channels = self.samples[idx]
        ims = [self.read_im(im) for im in channels]
        # Hack to get around the encoder having 3 layers, since we aren't tuning a custom one for this
        image_tensor = torch.concat([ims[0], ims[0], ims[1]], dim=0) 

        image_tensor = normalize_scale_for_test(image_tensor) # force to 224x224, self-norm is in passed transforms
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, self.class_map[self.labels[idx]]