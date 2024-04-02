import torch
from PIL import Image
import open_clip
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from binary_focal_loss import BinaryFocalLoss
from open_clip.model import get_cast_dtype
from open_clip.utils.env import checkpoint_pathmgr as pathmgr
import json
import sys
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from datasets.mvtec_dataset import mvtec_dataset, OBJECT_TYPE

state_level = {
               "normal":["{}", "flawless {}", "perfect {}", "unblemished {}",
                         "{} without flaw", "{} without defect", "{} without damage"],
                "anomaly":["damaged {}", "{} with flaw", "{} with defect", "{} with damage"]
}
template_level = [
                  "a cropped photo of the {}.",
                  "a cropped photo of a {}.",
                  "a close-up photo of a {}.",
                  "a close-up photo of the {}.",
                  "a bright photo of a {}.",
                  "a bright photo of the {}.",
                  "a dark photo of a {}.",
                  "a dark photo of the {}.",
                  "a jpeg corrupted photo of a {}.",
                  "a jpeg corrupted photo of the {}.",
                  "a blurry photo of the {}.",
                  "a blurry photo of a {}.",
                  "a photo of the {}.",
                  "a photo of a {}.",
                  "a photo of a small {}.",
                  "a photo of the small {}.",
                  "a photo of a large {}.",
                  "a photo of the large {}.",
                  "a photo of a {} for visual inspection.",
                  "a photo of the {} for visual inspection.",
                  "a photo of a {} for anomaly detection.",
                  "a photo of the {} for anomaly detection."
]

def get_texts(obj_name):
    normal_states = [s.format(obj_name) for s in state_level["normal"]]
    anomaly_states = [s.format(obj_name) for s in state_level["anomaly"]]

    normal_texts = [t.format(state) for state in normal_states for t in template_level]
    anomaly_texts = [t.format(state) for state in anomaly_states for t in template_level]

    return normal_texts, anomaly_texts

def run(config):
    # load model, preprocessor and tokenizer
    device = torch.cuda.current_device()
    tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')
    _, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240')

    cf = './open_clip/model_configs/ViT-B-16-plus-240.json'
    with open(cf, 'r') as f:
        model_cfg = json.load(f)
    embed_dim = model_cfg["embed_dim"]
    vision_cfg = model_cfg["vision_cfg"]
    text_cfg = model_cfg["text_cfg"]
    cast_dtype = get_cast_dtype('fp32')
    quick_gelu = False

    model = open_clip.model.WinCLIP(embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype)
    model = model.cuda(device=device)

    with pathmgr.open("./vit_b_16_plus_240-laion400m_e31-8fb26589.pt", "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    model.load_state_dict(checkpoint, strict=False)

    # load dataset
    obj_type = config['obj_type']
    shot = config["shot"]
    dataset = mvtec_dataset(config, config["data_dir"], mode='test', shot=shot, preprocess=preprocess)
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=8, shuffle=False)

    # load text template
    normal_texts, anomaly_texts = get_texts(obj_type.replace('_', " "))

    score_list = []
    gt_list = []
    for data in tqdm(dataloader, desc="Eval: ", total=len(dataloader)):
        image, ref_list, mask, has_anomaly, indice = data

        pos_features = tokenizer(normal_texts).cuda(device=device)
        neg_features = tokenizer(anomaly_texts).cuda(device=device)
        pos_features = model.encode_text(pos_features)
        neg_features = model.encode_text(neg_features)
        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)
        pos_features = torch.mean(pos_features, dim=0, keepdim=True)
        neg_features = torch.mean(neg_features, dim=0, keepdim=True)
        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)
        text_features = torch.cat([pos_features, neg_features], dim=0)

        if isinstance(image, list):
            pred = []
            for i in range(len(image)):
                img = image[i]
                img = img.cuda(device=device)
                _, _, image_features = model.encode_image(img)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                score = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                score = score[0, 1].cpu().numpy()
                pred.append(score)
            text_probs = sum(pred) / len(pred)
        else:
            image = image.cuda(device=device)
            _, _, image_features = model.encode_image(image)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            text_probs = text_probs[0, 1].cpu().numpy()

        if shot == 0:
            # zero-shot classification score
            score = text_probs
            score_list.append(score)
            gt_list.append(has_anomaly[0].numpy())
        else:
            # few-shot classification score
            img = []
            img.append(image)
            for i in ref_list:
                img.append(i)

            vis_probs = model.forward(shot=shot, image=img)
            score = (vis_probs + text_probs)/2
            score_list.append(score)
            gt_list.append(has_anomaly[0].numpy())

    auroc = roc_auc_score(gt_list, score_list)
    precision, recall, _ = precision_recall_curve(gt_list, score_list)
    aupr = auc(recall, precision)
    f1_max = 0
    for threshold in np.arange(0, 1, 0.01):
        y_pred = (score_list > threshold).astype(int)
        f1 = f1_score(gt_list, y_pred)
        if f1 > f1_max:
            f1_max = f1

    return gt_list, score_list, auroc, aupr, f1_max

if __name__ == "__main__":
    np.random.seed(10)
    torch.manual_seed(10)
    dataset_root_dir = '/home/zhujiawen/PDA/data/visa_anomaly_detection'
    datasetname = "visa"
    obj_type_id = 0
    config = {
        'datasetname': datasetname,
        'dataset_root_dir': dataset_root_dir,
        'data_dir': os.path.join(dataset_root_dir, datasetname),
        'obj_type_id': obj_type_id,
        'obj_type': OBJECT_TYPE[obj_type_id],
        'shot': 2,
    }
    all_auroc_list = []
    all_aupr_list = []
    all_f1_list = []
    all_gt_list = []
    all_score_list = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for obj_type in OBJECT_TYPE[:-1]:
            config['obj_type'] = obj_type
            gt_list, score_list, auroc, aupr, f1_max = run(config)
            all_auroc_list.append(auroc)
            all_aupr_list.append(aupr)
            all_f1_list.append(f1_max)
            all_gt_list += gt_list
            all_score_list += score_list
            print("Obj Type: {}, AUROC={}, AUPR={}, F1-Max={}".format(obj_type, auroc, aupr, f1_max))
    print('Avg auroc: {}'.format(np.mean(all_auroc_list)))
    print('Avg aupr: {}'.format(np.mean(all_aupr_list)))
    print('Avg f1-max: {}'.format(np.mean(all_f1_list)))

    auroc = roc_auc_score(all_gt_list, all_score_list)
    precision, recall, _ = precision_recall_curve(all_gt_list, all_score_list)
    aupr = auc(recall, precision)
    f1_max = 0
    for threshold in np.arange(0, 1, 0.01):
        y_pred = (all_score_list > threshold).astype(int)
        f1 = f1_score(all_gt_list, y_pred)
        if f1 > f1_max:
            f1_max = f1
    print("All Type: AUROC={}, AUPR={}, F1-Max={}".format(auroc, aupr, f1_max))