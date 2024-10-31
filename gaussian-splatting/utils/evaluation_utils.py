from PIL import Image
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
import torchvision
import os
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips


def update_json(identifier, scores, keys):
    """
    Update the evaluation_results.json file with the given identifier, scores, and keys.
    
    Args:
        identifier (str): The identifier for the data. Key for the json file.
        scores (list): The list of scores to be updated.
        keys (list): The list of keys corresponding to the scores.
    
    Returns:
        None
    """
    file_path = 'evaluation_results.json'
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump({}, f)
    with open(file_path, 'r') as f:
        data = json.load(f)
    data[identifier] = {}
    for i, key in enumerate(keys):
        data[identifier][key] = scores[i]
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return None


def process_pair(gt_mask_path, pred_mask_path, threshold=0.5, debug=False, name='pred'):
    """
    Process a pair of ground truth and predicted masks and calculate the IoU score and accuracy.

    Args:
        gt_mask_path (str): Path to the ground truth mask image file.
        pred_mask_path (str): Path to the predicted mask image file.
        threshold (float, optional): Threshold value for binarizing the predicted mask. 
            Defaults to 0.5.
        debug (bool, optional): Flag indicating whether to save debug images. Defaults to False.
        name (str, optional): Name for the predicted mask. Defaults to 'pred'.

    Returns:
        tuple: A tuple containing the IoU score and accuracy.
    """
    
    threshold = int(255 * threshold)
    box_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(pred_mask_path))))
    box_path = os.path.join(box_path, 'results_boxes')
    os.makedirs(box_path, exist_ok=True)
    gt_mask = np.array(Image.open(gt_mask_path))
    pred_mask = np.array(Image.open(pred_mask_path))
    gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
    pred_mask = pred_mask > threshold
    pred_mask = pred_mask[:, :, 0].astype(int) * 1
    pred_mask_save = torch.from_numpy(pred_mask).unsqueeze(0).cuda().contiguous().float()
    box_gt_path = os.path.join(box_path, '{}_mask.png'.format(name))
    torchvision.utils.save_image(pred_mask_save, box_gt_path)
    torchvision.utils.save_image(torch.from_numpy(gt_mask).float(), os.path.join(box_path, 'gt.png'))
    if np.amax(gt_mask) > 1:
        gt_mask = gt_mask / 255.

    if debug:
        plt.imshow(gt_mask, cmap='gray')
        plt.axis('off')
        plt.savefig('eval_gt.png')
        plt.imshow(pred_mask, cmap='gray')
        plt.axis('off')
        plt.savefig('eval_pred.png')
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    iou_score = np.sum(intersection) / np.sum(union)

    accuracy = np.mean(gt_mask == pred_mask)
    return iou_score, accuracy


def evaluate_spinnerf(model_path, identifier, gt_mask_path, final_mask_thresh):
    """
    Evaluate the performance on the spinnerf benchmark.

    Args:
        model_path (str): The path to the directory with the predicted masks.
        identifier (str): The identifier of the model.
        gt_mask_path (str): The path to the directory containing the ground truth masks.
        final_mask_thresh (float): The threshold for the final mask.

    Returns:
        None
    """
    coarse_mask_path = os.path.join(model_path, 'coarse_{}'.format(identifier),
                                    "select", "masks")
    fine_mask_path = os.path.join(model_path, 'fine_gc_{}'.format(identifier),
                                  "select", "masks")
    assert os.path.exists(coarse_mask_path), "Coarse mask path does not exist"
    assert os.path.exists(fine_mask_path), "Fine mask path does not exist"
    all_gt_files = os.listdir(gt_mask_path)
    remove_words = ['pseudo', 'cutout']
    all_gt_masks = []
    for img in all_gt_files:
        if img.endswith('.png'):
            if remove_words[0] in img or remove_words[1] in img:
                continue
            all_gt_masks.append(img)
    all_gt_masks.sort()

    all_coarse_masks = os.listdir(coarse_mask_path)
    all_coarse_masks.sort()
    all_fine_masks = os.listdir(fine_mask_path)
    all_fine_masks.sort()
    print('gt, coarse mask, fine mask:', len(all_gt_masks),
          len(all_coarse_masks), len(all_fine_masks))
    ious_coarse = []
    acc_coarse = []
    ious_fine = []
    acc_fine = []
    if len(all_gt_masks) != len(all_coarse_masks):
        # they should be the same name so we can match them
        for gt_mask in all_gt_masks:
            gt_base = gt_mask.split('.')[0]
            match_found = False
            for coarse_mask in all_coarse_masks:
                if coarse_mask.split('.')[0] == gt_base:
                    iou, acc = process_pair(
                        os.path.join(gt_mask_path, gt_mask),
                        os.path.join(coarse_mask_path, coarse_mask),
                        name='coarse', threshold=final_mask_thresh)
                    ious_coarse.append(iou)
                    acc_coarse.append(acc)
                    match_found = True
                    break
                if not match_found:
                    ValueError(f"No match found for coarse {gt_mask}")
            match_found = False
            for fine_mask in all_fine_masks:
                if fine_mask.split('.')[0] == gt_base:
                    iou, acc = process_pair(
                        os.path.join(gt_mask_path, gt_mask),
                        os.path.join(fine_mask_path, fine_mask), name='fine',
                        threshold=final_mask_thresh)
                    ious_fine.append(iou)
                    acc_fine.append(acc)
                    match_found = True
                    break
                if not match_found:
                    ValueError(f"No match found for fine {gt_mask}")
    else:
        for gt_index in range(len(all_gt_masks)):
            iou, acc = process_pair(
                os.path.join(gt_mask_path, all_gt_masks[gt_index]),
                os.path.join(coarse_mask_path, all_coarse_masks[gt_index]),
                name='coarse', threshold=final_mask_thresh)
            ious_coarse.append(iou)
            acc_coarse.append(acc)
            iou, acc = process_pair(
                os.path.join(gt_mask_path, all_gt_masks[gt_index]),
                os.path.join(fine_mask_path, all_fine_masks[gt_index]),
                name='fine', threshold=final_mask_thresh)
            ious_fine.append(iou)
            acc_fine.append(acc)
    print('Coarse iou, acc:', np.mean(ious_coarse), np.mean(acc_coarse))
    print('Fine iou, acc:', np.mean(ious_fine), np.mean(acc_fine))
    scores_json = [
        np.mean(ious_coarse),
        np.mean(acc_coarse),
        np.mean(ious_fine),
        np.mean(acc_fine)
    ]
    keys_json = ['iou_coarse', 'acc_coarse', 'iou_fine', 'acc_fine']
    update_json(identifier, scores_json, keys_json)

    return None


def save_boxes(gt_mask_path, image_path, coarse_image_path, fine_image_path,
               debug=False):
    """
    Saves the cropped ground truth, fine prediction, and coarse prediction images
    along with their corresponding image quality scores.

    Args:
        gt_mask_path (str): Path to the ground truth mask image.
        image_path (str): Path to the original image.
        coarse_image_path (str): Path to the coarse prediction image.
        fine_image_path (str): Path to the fine prediction image.
        debug (bool, optional): Whether to save debug images. Defaults to False.

    Returns:
        list: List of evaluation scores including SSIM, PSNR, and LPIPS.
    """
    box_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(coarse_image_path))))
    box_path = os.path.join(box_path, 'results_boxes')
    os.makedirs(box_path, exist_ok=True)

    gt_mask = np.array(Image.open(gt_mask_path))
    gt_image = np.array(Image.open(image_path))
    coarse_image = np.array(Image.open(coarse_image_path))
    fine_image = np.array(Image.open(fine_image_path))
    gt_mask = cv2.resize(gt_mask, (gt_image.shape[1], gt_image.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    gt_image[gt_mask == 0] = [0, 0, 0]
    rows = np.any(gt_mask, axis=1)
    cols = np.any(gt_mask, axis=0)

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    cropped_gt = gt_image[ymin:ymax, xmin:xmax]
    cropped_pred = fine_image[ymin:ymax, xmin:xmax]
    cropped_pred_coarse = coarse_image[ymin:ymax, xmin:xmax]
    if debug:
        plt.imshow(cropped_gt)
        plt.axis('off')
        plt.savefig('eval_gt.png')

        plt.imshow(cropped_pred)
        plt.axis('off')
        plt.savefig('eval_pred_fine.png')

        plt.imshow(cropped_pred_coarse)
        plt.axis('off')
        plt.savefig('eval_pred_coarse.png')

    box_gt_path = os.path.join(box_path, 'gt.png')
    box_pred_path = os.path.join(box_path, 'pred_fine.png')
    box_pred_coarse_path = os.path.join(box_path, 'pred_coarse.png')
    cropped_gt = torch.from_numpy(cropped_gt / 255.).permute(
        2, 0, 1).unsqueeze(0)[:, :3, :, :].cuda().contiguous().float()
    cropped_pred = torch.from_numpy(cropped_pred / 255.).permute(
        2, 0, 1).unsqueeze(0)[:, :3, :, :].cuda().contiguous().float()
    cropped_pred_coarse = torch.from_numpy(cropped_pred_coarse / 255.).permute(
        2, 0, 1).unsqueeze(0)[:, :3, :, :].cuda().contiguous().float()
    torchvision.utils.save_image(cropped_gt, box_gt_path)
    torchvision.utils.save_image(cropped_pred, box_pred_path)
    torchvision.utils.save_image(cropped_pred_coarse, box_pred_coarse_path)
    ssim_score_coarse = ssim(cropped_pred_coarse, cropped_gt)
    ssim_score_fine = ssim(cropped_pred, cropped_gt)
    psnr_score_coarse = psnr(cropped_pred_coarse, cropped_gt)
    psnr_score_fine = psnr(cropped_pred, cropped_gt)
    lpips_score_coarse = lpips(cropped_pred_coarse, cropped_gt, net_type='vgg')
    lpips_score_fine = lpips(cropped_pred, cropped_gt, net_type='vgg')
    scores = [
        ssim_score_coarse, ssim_score_fine, psnr_score_coarse, psnr_score_fine,
        lpips_score_coarse, lpips_score_fine
    ]
    for score_index in range(len(scores)):
        scores[score_index] = scores[score_index].item()
    return scores


def evaluate_nvos(model_path, identifier, gt_mask_path, final_mask_thresh):
    coarse_mask_path = os.path.join(model_path, 'coarse_{}'.format(identifier),
                                    "select", "masks")
    fine_mask_path = os.path.join(model_path, 'fine_gc_{}'.format(identifier),
                                  "select", "masks")
    coarse_image_path = os.path.join(model_path, 'coarse_{}'.format(identifier),
                                     "select", "renders")
    fine_image_path = os.path.join(model_path, 'fine_gc_{}'.format(identifier),
                                   "select", "renders")
    gt_image_path = os.path.join(model_path, 'coarse_{}'.format(identifier),
                                 "select", "gt")
    assert os.path.exists(coarse_mask_path), "Coarse mask path does not exist"
    assert os.path.exists(fine_mask_path), "Fine mask path does not exist"
    all_gt_files = os.listdir(gt_mask_path)
    all_gt_files.sort()
    all_coarse_masks = os.listdir(coarse_mask_path)
    all_coarse_masks.sort()
    all_fine_masks = os.listdir(fine_mask_path)
    all_fine_masks.sort()
    all_coarse_images = os.listdir(coarse_image_path)
    all_coarse_images.sort()
    all_fine_images = os.listdir(fine_image_path)
    all_fine_images.sort()
    all_gt_images = os.listdir(gt_image_path)
    all_gt_images.sort()
    all_gt_masks = []
    for img in all_gt_files:
        if 'mask' in img:
            all_gt_masks.append(img)
    all_gt_masks.sort()
    print('gt, coarse mask, fine mask:', len(all_gt_masks),
          len(all_coarse_masks), len(all_fine_masks))
    ious_coarse = []
    acc_coarse = []
    ious_fine = []
    acc_fine = []
    all_psnr_coarse = []
    all_psnr_fine = []
    all_ssim_coarse = []
    all_ssim_fine = []
    all_lpips_coarse = []
    all_lpips_fine = []
    assert len(all_gt_masks) == len(all_coarse_masks) == len(
        all_fine_masks) == len(all_coarse_images) == len(
            all_fine_images), "Number of masks and images do not match"
    for gt_index in range(len(all_gt_masks)):
        iou, acc = process_pair(
            os.path.join(gt_mask_path, all_gt_masks[gt_index]),
            os.path.join(coarse_mask_path, all_coarse_masks[gt_index]),
            name='coarse', threshold=final_mask_thresh)
        ious_coarse.append(iou)
        acc_coarse.append(acc)
        iou, acc = process_pair(
            os.path.join(gt_mask_path, all_gt_masks[gt_index]),
            os.path.join(fine_mask_path, all_fine_masks[gt_index]), name='fine',
            threshold=final_mask_thresh)
        ious_fine.append(iou)
        acc_fine.append(acc)
        scores = save_boxes(
            os.path.join(gt_mask_path, all_gt_masks[gt_index]),
            os.path.join(gt_image_path, all_gt_images[gt_index]),
            os.path.join(coarse_image_path, all_coarse_images[gt_index]),
            os.path.join(fine_image_path, all_fine_images[gt_index]))
        all_ssim_coarse.append(scores[0])
        all_ssim_fine.append(scores[1])
        all_psnr_coarse.append(scores[2])
        all_psnr_fine.append(scores[3])
        all_lpips_coarse.append(scores[4])
        all_lpips_fine.append(scores[5])

    print('Coarse iou, acc:', np.mean(ious_coarse), np.mean(acc_coarse))
    print('Fine iou, acc:', np.mean(ious_fine), np.mean(acc_fine))
    print('Coarse ssim, psnr, lpips:', np.mean(all_ssim_coarse),
          np.mean(all_psnr_coarse), np.mean(all_lpips_coarse))
    print('Fine ssim, psnr, lpips:', np.mean(all_ssim_fine),
          np.mean(all_psnr_fine), np.mean(all_lpips_fine))
    scores_json = [
        np.mean(ious_coarse),
        np.mean(acc_coarse),
        np.mean(ious_fine),
        np.mean(acc_fine),
        np.mean(all_ssim_coarse),
        np.mean(all_ssim_fine),
        np.mean(all_psnr_coarse),
        np.mean(all_psnr_fine),
        np.mean(all_lpips_coarse),
        np.mean(all_lpips_fine)
    ]
    keys_json = [
        'iou_coarse', 'acc_coarse', 'iou_fine', 'acc_fine', 'ssim_coarse',
        'ssim_fine', 'psnr_coarse', 'psnr_fine', 'lpips_coarse', 'lpips_fine'
    ]
    update_json(identifier, scores_json, keys_json)
    return None
