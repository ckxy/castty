import math
import torch
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_mask(img, mask, classes, colorbar=True, erase_contour=False, have_background=True):
    uni = np.unique(mask).tolist()
    have_contour = 255 in uni

    if have_contour and not erase_contour:
        num_classes = len(classes)
    else:
        num_classes = len(classes) - 1
    
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    colors = [(0, 0, 0)] + colors

    if have_contour and not erase_contour:
        tmp = colors.pop(-1)
        for i in range(255 - len(colors)):
            colors.append((0, 0, 0))
        colors.append(tmp)
    elif have_contour and erase_contour:
        mask = np.array(mask).astype(np.int)
        mask[mask == 255] = 0
        mask = Image.fromarray(mask.astype(np.uint8))

    mask.putpalette(np.array(colors).flatten().tolist())
    mask = mask.convert("RGB")

    img = Image.blend(img, mask, 0.5)

    if colorbar:
        if have_background and 0 in uni:
            uni.pop(0)
        if len(uni) > 0:
            colors = [colors[u] for u in uni]
            classes = [classes[u] if u != 255 else 'boundary' for u in uni]

            w, h = img.size
            w = w // 4

            bar = Image.new('RGB', (w, h), (255, 255, 255))
            l = math.sqrt(h * h + w * w)
            draw = ImageDraw.Draw(bar)
            font = ImageFont.truetype("fonts/arial.ttf", int(l * 5e-2))

            pw = w
            ph = h // len(colors)

            x1, y1 = 0, (h - ph * len(colors)) // 2
            for i in range(len(colors)):
                draw.rectangle((x1, y1, x1 + pw, y1 + ph), fill=colors[i], outline=(0, 0, 0))
                draw.text((x1, y1), classes[i], fill=(0, 0, 0), font=font)
                y1 += ph
        else:
            w, h = img.size
            w = w // 4

            bar = Image.new('RGB', (w, h), (255, 255, 255))
            l = math.sqrt(h * h + w * w)
            draw = ImageDraw.Draw(bar)
            font = ImageFont.truetype("fonts/arial.ttf", int(l * 5e-2))
            draw.text((0, 0), 'no_label', fill=(0, 0, 0), font=font)

        return img, bar
    else:
        return img, None


def pixel_accuracy(imPred, imLab):
    """
    This function takes the prediction and label of a single image, returns pixel-wise accuracy
    To compute over many images do:
    for i = range(Nimages):
         (pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = \
            pixelAccuracy(imPred[i], imLab[i])
    mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return pixel_accuracy, pixel_correct, pixel_labeled


def intersection_union(imPred, imLab, numClass):
    """
    This function takes the prediction and label of a single image,
    returns intersection and union areas for each class
    To compute over many images do:
    for i in range(Nimages):
        (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab >= 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return area_intersection, area_union


# pytorch version
def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    # predict = torch.argmax(output.long(), 1) + 1
    predict = output.long() + 1
    target = target.long() + 1

    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    # predict = torch.argmax(output, 1) + 1
    predict = output + 1
    target = target.float() + 1

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()
