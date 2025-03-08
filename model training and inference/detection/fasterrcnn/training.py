import torch
import argparse
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.ops.boxes import box_iou
import os
import csv
import numpy as np
from data_loader import WCEBleedGenDataset, collate_fn

def get_dataloaders(dataset_root, batch_size, num_workers):
    transform = T.Compose([T.ToTensor()])
    train_dataset = WCEBleedGenDataset(root=dataset_root, split="train", transform=transform, mask_transform=transform)
    val_dataset = WCEBleedGenDataset(root=dataset_root, split="val", transform=transform, mask_transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
    return train_loader, val_loader

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(args.dataset_root, args.batch_size, args.num_workers)
    
    model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    csv_file = open("training_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "precision", "recall", "map_50"])
    
    best_map = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        model.eval()
        val_loss = 0
        all_ap = []
        tp_count, fp_count, fn_count = 0, 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                outputs = model(images)
                for output, target in zip(outputs, targets):
                    pred_boxes, gt_boxes = output['boxes'], target['boxes']
                    
                    if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                        ious = box_iou(pred_boxes, gt_boxes)
                        matches = ious.max(dim=1)[0] > 0.5
                        all_ap.append(matches.float().mean().item())
                    
                    if len(gt_boxes) > 0:
                        if len(pred_boxes) > 0:
                            tp_count += 1
                        else:
                            fn_count += 1
                    if len(gt_boxes) == 0 and len(pred_boxes) > 0:
                        fp_count += 1
        
        map_50 = np.mean(all_ap) if all_ap else 0.0
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, mAP@0.5 {map_50:.4f}")
        csv_writer.writerow([epoch+1, train_loss, val_loss, precision, recall, map_50])
        csv_file.flush()
        
        if map_50 > best_map:
            best_map = map_50
            torch.save(model.state_dict(), "best.pth")
            print(f"Epoch {epoch+1}: New best model saved with mAP@0.5 {best_map:.4f}")
    
    csv_file.close()
    torch.save(model, "full_faster_rcnn.pth")
    print("Full model saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--step_size", type=int, default=5, help="Scheduler step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="Scheduler decay factor")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()
    
    main(args)
