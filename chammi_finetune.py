from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
from lightning import Fabric
from torch import nn, zeros_like
from torch.optim import AdamW
from torch import set_float32_matmul_precision
from lightning.fabric.strategies import DDPStrategy
set_float32_matmul_precision('high')
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model
import argparse
from dataset import * # am lazy, T_T
from torchvision import transforms
import torch

class CHAMMIClassifier(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.predictor = nn.Linear(embed_dim, num_classes)
        self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()

    def forward(self, batch):
        outputs = self.backbone(batch).pooler_output
        logits = self.predictor(outputs)
        return logits

from peft import LoraConfig, get_peft_model
def peft_backbone(backbone):    
    lora_params = ["k_proj", "v_proj", "q_proj", "o_proj", "up_proj", "down_proj"]
    config = LoraConfig(
        r=256,
        target_modules=lora_params,
    )

    backbone = get_peft_model(backbone, config, autocast_adapter_dtype=False)

    for name, param in backbone.named_parameters():
        if not any(layer in name for layer in lora_params):
            param.requires_grad = True
    return backbone

def main():
    model_name, data_path, metadata_path = parse_args()
    
    transform = transforms.Compose([transforms.ConvertImageDtype(torch.float32), self_normalize()])
    dataset = MultiChannelDataset(
        metadata=metadata_path,
        im_dir=data_path,
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=256, num_workers=8, pin_memory=True, shuffle=True)
       
    pretrained_model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    # pretrained_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    backbone = AutoModel.from_pretrained(pretrained_model_name, device_map="auto")
    # backbone = peft_backbone(backbone)
    # For from scratch through huggingface
    # backbone_config = AutoConfig.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m", device_map="auto")
    # backbone = AutoModel.from_config(backbone_config)
    
    model = CHAMMIClassifier(backbone, 768, dataset.num_classes)
    optim = AdamW(model.parameters(), lr=0.0001)
    strategy = DDPStrategy(find_unused_parameters=True)
    fabric = Fabric(strategy=strategy)
    
    model, optim = fabric.setup(model, optim)
    train_dataloader = fabric.setup_dataloaders(loader)

    if fabric.global_rank == 0:
        wandb.init(
                project="dinov3_chammi",
                name=model_name,
                # mode="disabled"
            )
    loss_fn = nn.CrossEntropyLoss()
    

    it = 0
    num_epochs = 100
    
    model.freeze_backbone()
    for epoch in range(num_epochs):
        epoch_dataloader = tqdm(train_dataloader, desc=f"Epoch: {epoch}", total=len(train_dataloader), disable=not fabric.is_global_zero)
        
        if epoch == 1:
            model.unfreeze_backbone()
        for _, (bat, labels) in enumerate(epoch_dataloader):
            optim.zero_grad()
            pred = model.forward(bat)
            loss = loss_fn(pred, labels)
            if it % 20 == 0 and fabric.global_rank == 0: 
                wandb.log({'loss': loss.item()}, step=it)
            fabric.backward(loss)
            optim.step()
            it += 1
            
        if epoch % 10 == 0:
            state = {"model": model}
            save_path = f"/hdd/jcaicedo/projects/dinov3/ddw2026/models/{model_name}/eval/checkpoint_{epoch}.pt"
            if fabric.is_global_zero and not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fabric.save(save_path, state)
        fabric.barrier()
            
    state = {"model": model}
            
    save_path = f"/hdd/jcaicedo/projects/dinov3/ddw2026/models/{model_name}/eval/checkpoint_{num_epochs}.pt"
    if fabric.is_global_zero:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fabric.save(save_path, state)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        help="The model name for directory creation and wandb logging.",
        required=True,
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        help="The path to the dataset directory.",
        required=True,
    )
    
    parser.add_argument(
        "--metadata-path",
        type=str,
        help="The path to the metadata file.",
        required=True,
    )

    return parser

def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    
    args.data_path = path_expansion(args.data_path)
    args.metadata_path = path_expansion(args.metadata_path)
        
    return args.model_name, args.data_path, args.metadata_path

def path_expansion(path: str):
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))

if __name__ == "__main__":
    main()    

    