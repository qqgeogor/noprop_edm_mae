import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from pathlib import Path
from tqdm import tqdm
from train_mae_cifar10_noprop import MaskedDenoisingVit
from torch.cuda.amp import autocast, GradScaler
import mlflow
import mlflow.pytorch

class CifarClassifier(nn.Module):
    def __init__(self, backbone, num_classes=10,freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.embed_dim,num_classes)
        )
        
    def forward(self, x):
        # Get features from the backbone
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone.forward_feature(x)[:,0,:]
        else:
            features = self.backbone.forward_feature(x)[:,0,:]
        
        # Pass through classifier
        return self.classifier(features)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=args.run_name):
        # Log parameters
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "model_type": "ViT",
            "embed_dim": args.embed_dim,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "freeze_backbone": args.freeze_backbone,
            "img_size": args.img_size,
            "patch_size": args.patch_size,
            "use_amp": args.use_amp
        })
        
        # Initialize GradScaler for AMP
        scaler = GradScaler(enabled=args.use_amp)

        # Data preprocessing for training
        train_transform = transforms.Compose([
            transforms.Resize(int(args.img_size*1.14),interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(args.img_size,padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Data preprocessing for testing
        test_transform = transforms.Compose([
            transforms.Resize(args.img_size,interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load CIFAR-10
        trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                              download=True, transform=train_transform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers)
        
        testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                             download=True, transform=test_transform)
        testloader = DataLoader(testset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
        
        trainer = MaskedDenoisingVit(T=args.T,args=args,device=device,trainset=trainset,train_loader=trainloader)
        
        if args.pretrained_path:
            trainer.load_checkpoint(args.pretrained_path)
        else:
            print("Pretrained path is required")
        
        backbone = trainer.encoder
        # Create classifier
        model = CifarClassifier(backbone,freeze_backbone=args.freeze_backbone).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        best_acc = 0
        
        # Training loop
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            
            for i, (images, labels) in enumerate(tqdm(trainloader)):
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass with autocast
                with autocast(enabled=args.use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # Backward pass with scaler
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item()
                
                if i % args.log_freq == 0:
                    print(f'Epoch [{epoch}/{args.epochs}], Step [{i}/{len(trainloader)}], '
                          f'Loss: {loss.item():.4f}')
            
            # Calculate average loss for the epoch
            avg_loss = running_loss / len(trainloader)
            
            # Evaluate (no need for autocast in eval mode)
            test_acc = evaluate(model, testloader, device)
            print(f'Epoch [{epoch}/{args.epochs}], Test Accuracy: {test_acc:.2f}%')
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "test_accuracy": test_acc
            }, step=epoch)
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                checkpoint_path = os.path.join(args.output_dir, 'best_classifier.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'accuracy': test_acc,
                }, checkpoint_path)
                
                # Log best model to MLflow
                mlflow.pytorch.log_model(model, "best_model")
                mlflow.log_artifact(checkpoint_path)
            
            # Save checkpoint
            if epoch % args.save_freq == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),  # Save scaler state
                    'accuracy': test_acc,
                }, os.path.join(args.output_dir, f'classifier_checkpoint_{epoch}.pth'))

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser('CIFAR-10 Classifier Finetuning')
    
    # Training parameters
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--encoder_type', default='student', type=str)
    parser.add_argument('--freeze_backbone', action='store_true',default=False,
                       help='Freeze backbone during training')
    # System parameters
    parser.add_argument('--data_path', default='c:/dataset', type=str)
    parser.add_argument('--output_dir', default='F:/output/cifar10-classifier')
    parser.add_argument('--pretrained_path', default='', 
                       type=str, help='Path to pretrained SimSiam model')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--T', default=4, type=int)
    
    # device
    parser.add_argument('--device',default='cuda',type=str)

    # model parameters
    parser.add_argument('--img_size', default=128, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--embed_dim', default=192, type=int)
    parser.add_argument('--depth', default=12, type=int)
    parser.add_argument('--num_heads', default=3, type=int)
    parser.add_argument('--decoder_embed_dim', default=96, type=int)
    parser.add_argument('--decoder_depth', default=4, type=int)
    parser.add_argument('--decoder_num_heads', default=3, type=int)
    parser.add_argument('--mlp_ratio', default=4., type=float)
    parser.add_argument('--model_type', default='student', type=str)
    parser.add_argument('--use_checkpoint', action='store_true',default=False,
                       help='Use checkpoint during training')
    
    
    # Add optimizer arguments
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=[0.5, 0.999], type=float, nargs='+',
                        help='Optimizer Betas (default: [0.9, 0.999])')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--linear_schedule', action='store_true',
                        help='Use linear schedule for alpha')
    

    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help='Number of epochs for warmup')
    parser.add_argument('--start_warmup_value', default=1e-6, type=float,
                        help='Start warmup value')
    
    
    # Add MLflow arguments
    parser.add_argument('--mlflow_tracking_uri', default='http://localhost:5000',
                      help='URI for MLflow tracking server')
    parser.add_argument('--mlflow_experiment_name', default='cifar10_classifier',
                      help='MLflow experiment name')
    parser.add_argument('--run_name', default=None, type=str,
                      help='Name for the MLflow run')
    
    # Add AMP argument
    parser.add_argument('--use_amp', action='store_true',
                       help='Use Automatic Mixed Precision training')
    
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    finetune(args) 