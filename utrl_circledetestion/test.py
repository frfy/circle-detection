# Compute the prediction with ONNX Runtime
import onnxruntime as rt
import numpy

import numpy as np
import torch
import time
import random

import os

import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn.functional as F

import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)

from cython_nms import nms as cnms

# 参数
_target_size = 275

_prior_path = "priors_" + str(_target_size) + ".txt"  # 转onnx时，预设框有问题，所以提前保存下来，推断时用

_undo_transform = False  # 1
_mask_alpha = 0.45

# 后处理
_display_lincomb = False
_crop = True
_score_threshold = 0.5

_top_k = 1
_display_masks = True
_eval_mask_branch = True
_display_scores = True
_mask_alpah = 0.45
_eval_mask_branch = True

# preprocess
_means = (103.94, 116.78, 123.68)
_std = (57.38, 57.12, 58.40)
_normalize = True
_subtract_means = False
_to_float = False
_channel_order = "RGB"

# detectonnx
_num_classes = 2
_bkg_label = 0  # 背景标签默认为0
_conf_thresh = 0.5
_nms_thresh = 0.5
_max_num_detections = 100

# 后处理
# mask_type
_mask_type = 1
_direct = 0
_lincomb = 1
_mask_proto_mask_activation = torch.sigmoid

# 显示
_display_text = True
_display_bboxes = True
COLORS = ((244, 67, 54),
          (233, 30, 99),
          (156, 39, 176),
          (103, 58, 183),
          (63, 81, 181),
          (33, 150, 243),
          (3, 169, 244),
          (0, 188, 212),
          (0, 150, 136),
          (76, 175, 80),
          (139, 195, 74),
          (205, 220, 57),
          (255, 235, 59),
          (255, 193, 7),
          (255, 152, 0),
          (255, 87, 34),
          (121, 85, 72),
          (158, 158, 158),
          (96, 125, 139))
_class = "circle"

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep
# 预处理
def preprocess(img):
    # torch tensor转numpy处理
    mean = torch.Tensor(_means).float()[None, :, None, None]
    std = torch.Tensor(_std).float()[None, :, None, None]

    img_size = (_target_size, _target_size)

    img = img.permute(0, 3, 1, 2).contiguous()
    img = F.interpolate(img, img_size, mode='bilinear', align_corners=False)

    if _normalize:
        img = (img - mean) / std
    elif _subtract_means:
        img = (img - mean)
    elif _to_float:
        img = img / 255

    if _channel_order != 'RGB':
        raise NotImplementedError

    img = img[:, (2, 1, 0), :, :].contiguous()

    # Return value is in channel order [n, c, h, w] and RGB
    return img


def decode(loc, priors, use_yolo_regressors: bool = False):
    """
    Decode predicted bbox coordinates using the same scheme
    employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf

        b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
        b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
        b_w = prior_w * exp(loc_w)
        b_h = prior_h * exp(loc_h)

    Note that loc is inputed as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
    while priors are inputed as [x, y, w, h] where each coordinate
    is relative to size of the image (even sigmoid(x)). We do this
    in the network by dividing by the 'cell size', which is just
    the size of the convouts.

    Also note that prior_x and prior_y are center coordinates which
    is why we have to subtract .5 from sigmoid(pred_x and pred_y).

    Args:
        - loc:    The predicted bounding boxes of size [num_priors, 4]
        - priors: The priorbox coords with size [num_priors, 4]

    Returns: A tensor of decoded relative coordinates in point form
             form with size [num_priors, 4]
    """

    if use_yolo_regressors:
        # Decoded boxes in center-size notation
        boxes = torch.cat((
            loc[:, :2] + priors[:, :2],
            priors[:, 2:] * torch.exp(loc[:, 2:])
        ), 1)
        ###########################
        # boxes = point_form(boxes)
    else:
        variances = [0.1, 0.2]

        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

    return boxes


def traditional_nms(boxes, masks, scores, iou_threshold, conf_thresh):
    num_classes = scores.size(0)

    idx_lst = []
    cls_lst = []
    scr_lst = []

    # Multiplying by max_size is necessary because of how cnms computes its area and intersections
    boxes = boxes * _target_size

    for _cls in range(num_classes):
        cls_scores = scores[_cls, :]
        conf_mask = cls_scores > conf_thresh

        #
        idx = torch.arange(cls_scores.size(0), device=boxes.device)

        cls_scores = cls_scores[conf_mask]
        idx = idx[conf_mask]

        if cls_scores.size(0) == 0:
            continue

        # torch
        preds = torch.cat([boxes[conf_mask], cls_scores[:, None]], dim=1).cpu().numpy()
        keep = cnms(preds, iou_threshold)
        keep = torch.Tensor(keep, device=boxes.device).long()

        idx_lst.append(idx[keep])
        cls_lst.append(keep * 0 + _cls)
        scr_lst.append(cls_scores[keep])

    idx = torch.cat(idx_lst, dim=0)
    classes = torch.cat(cls_lst, dim=0)
    scores = torch.cat(scr_lst, dim=0)

    scores, idx2 = scores.sort(0, descending=True)
    idx2 = idx2[:_max_num_detections]
    scores = scores[:_max_num_detections]

    idx = idx[idx2]
    classes = classes[idx2]

    # Undo the multiplication above
    return boxes[idx] / _target_size, masks[idx], classes, scores


def detect(batch_idx, conf_preds, decoded_boxes, mask_data, inst_data):
    """ Perform nms for only the max scoring class that isn't background (class 0) """
    cur_scores = conf_preds[batch_idx, 1:, :]
    conf_scores, _ = torch.max(cur_scores, dim=0)

    keep = (conf_scores > _conf_thresh)
    scores = cur_scores[:, keep]
    boxes = decoded_boxes[keep, :]
    masks = mask_data[batch_idx, keep, :]

    if inst_data is not None:
        inst = inst_data[batch_idx, keep, :]

    if scores.size(1) == 0:
        return None

    boxes, masks, classes, scores = traditional_nms(boxes, masks, scores, _nms_thresh, _conf_thresh)

    return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}


def detectonnx(predictions, num_classes, bkg_label=0, top_k=200, conf_thresh=0.5, nms_thresh=0.5):
    loc_data = predictions['loc']
    conf_data = predictions['conf']
    mask_data = predictions['mask']
    prior_data = predictions['priors']

    proto_data = predictions['proto'] if 'proto' in predictions else None
    inst_data = predictions['inst'] if 'inst' in predictions else None

    out = []

    batch_size = loc_data.size(0)
    num_priors = prior_data.size(0)

    conf_preds = conf_data.view(batch_size, num_priors, num_classes).transpose(2, 1).contiguous()

    for batch_idx in range(batch_size):
        decoded_boxes = decode(loc_data[batch_idx], prior_data)
        result = detect(batch_idx, conf_preds, decoded_boxes, mask_data, inst_data)

        if result is not None and proto_data is not None:
            result['proto'] = proto_data[batch_idx]

        out.append(result)

    return out


def sanitize_coordinates(_x1, _x2, img_size: int, padding: int = 0, cast: bool = True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1 - padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)

    return x1, x2


def crop(masks, boxes, padding: int = 1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask.float()


def postprocess(det_output, w, h, batch_idx=0, interpolation_mode='bilinear',
                visualize_lincomb=False, crop_masks=True, score_threshold=0):
    """
    Postprocesses the output of Yolact on testing mode into a format that makes sense,
    accounting for all the possible configuration settings.

    Args:
        - det_output: The lost of dicts that Detect outputs.
        - w: The real with of the image.
        - h: The real height of the image.
        - batch_idx: If you have multiple images for this batch, the image's index in the batch.
        - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)

    Returns 4 torch Tensors (in the following order):
        - classes [num_det]: The class idx for each detection.
        - scores  [num_det]: The confidence score for each detection.
        - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
        - masks   [num_det, h, w]: Full image masks for each detection.
    """

    dets = det_output[batch_idx]

    if dets is None:
        return [torch.Tensor()] * 4  # Warning, this is 4 copies of the same thing

    if score_threshold > 0:
        keep = dets['score'] > score_threshold

        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][keep]

        if dets['score'].size(0) == 0:
            return [torch.Tensor()] * 4

    # im_w and im_h when it concerns bboxes. This is a workaround hack for preserve_aspect_ratio
    b_w, b_h = (w, h)

    # Actually extract everything from dets now
    classes = dets['class']
    boxes = dets['box']
    scores = dets['score']
    masks = dets['mask']

    if _mask_type == _lincomb and _eval_mask_branch:
        # At this points masks is only the coefficients
        proto_data = dets['proto']

        masks = torch.matmul(proto_data, masks.t())
        masks = _mask_proto_mask_activation(masks)

        # Crop masks before upsampling because you know why
        if crop_masks:
            masks = crop(masks, boxes)

        # Permute into the correct output shape [num_dets, proto_h, proto_w]
        masks = masks.permute(2, 0, 1).contiguous()

        masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interpolation_mode, align_corners=False).squeeze(0)

        # Binarize the masks
        masks.gt_(0.5)

    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], b_w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], b_h, cast=False)
    boxes = boxes.long()
    if _mask_type == _direct and _eval_mask_branch:
        # Upscale masks
        full_masks = torch.zeros(masks.size(0), h, w)

        for jdx in range(masks.size(0)):
            x1, y1, x2, y2 = boxes[jdx, :]

            mask_w = x2 - x1
            mask_h = y2 - y1

            # Just in case
            if mask_w * mask_h <= 0 or mask_w < 0:
                continue
            ##############################################
            # mask = masks[jdx, :].view(1, 1, cfg.mask_size, cfg.mask_size)
            # mask = F.interpolate(mask, (mask_h, mask_w), mode=interpolation_mode, align_corners=False)
            # mask = mask.gt(0.5).float()
            # full_masks[jdx, y1:y2, x1:x2] = mask

        masks = full_masks
    print("masks.shape", masks.shape)
    return classes, scores, boxes, masks


def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """

    img_gpu = img / 255.0
    h, w, _ = img.shape

    t = postprocess(dets_out, w, h, visualize_lincomb=_display_lincomb,
                    crop_masks=_crop,
                    score_threshold=_score_threshold)

    if _eval_mask_branch:
        # Masks are drawn on the GPU, so don't copy
        masks = t[3][:_top_k]

    # boxes 框
    classes, scores, boxes = [x[:_top_k].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(_top_k, classes.shape[0])

    for j in range(num_dets_to_consider):
        if scores[j] < _score_threshold:
            num_dets_to_consider = j
            break

    if num_dets_to_consider == 0:
        # No detections found so just output the original image
        return (img_gpu * 255).byte().cpu().numpy()

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if _display_masks and _eval_mask_branch:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
        print("masks", masks.shape)
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat(
            [(torch.Tensor(get_color(j)).float() / 255).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]

        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * 0* inv_alph_masks.prod(dim=0) + masks_color_summand

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    # if _display_text or _display_bboxes:
    #     for j in reversed(range(num_dets_to_consider)):
    #         x1, y1, x2, y2 = boxes[j, :]
    #         color = get_color(j)
    #         score = scores[j]

            # if _display_bboxes:
            #     cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
            #
            # if _display_text:
            #     text_str = '%s: %.2f' % (_class, score) if _display_scores else _class
            #
            #     font_face = cv2.FONT_HERSHEY_DUPLEX
            #     font_scale = 0.6
            #     font_thickness = 1
            #
            #     text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
            #
            #     text_pt = (x1, y1 - 3)
            #     text_color = [255, 255, 255]
            #
            #     cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
            #     cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
            #                 cv2.LINE_AA)

    return img_numpy

def eucliDist(A, B):
    return np.sqrt(sum(np.power((A - B), 2)))


def detect_circle_by_hough(frame):
    height, width = frame.shape[0], frame.shape[1]
    assert height == width
    min_radus = int(height / 3*1.3)
    center_x, center_y = frame.shape[1] / 2.0, frame.shape[0] / 2.0

    output = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
    gray = cv2.GaussianBlur(gray, (5, 5), 0);
    gray = cv2.medianBlur(gray, 5)
    # Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.

    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                 cv2.THRESH_BINARY, 71, 3)
    kernel = np.ones((2, 2), np.uint8)
    gray = cv2.erode(gray, kernel, iterations=1)
    # gray = erosion

    gray = cv2.dilate(gray, kernel, iterations=1)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=10, param2=3, minRadius=min_radus,
                               maxRadius=height)
    """
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 0.1, 200, param1=90, param2=40, minRadius=min_radus,
                               maxRadius=height)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=60, param2=40, minRadius=min_radus,
                               maxRadius=height)
    """
    dists = []
    radiuss = []

    circle_output = []
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            dist = eucliDist(np.array([x, y]), np.array([center_x, center_y]))
            dists.append(dist)

        index = np.argmin(dist)

        x_, y_, r_ = circles[index]

        circle_output.extend(circles[index])

    return circle_output


def evalimage(imgname: str,output_dir:str,model_path:str,session):
    sess=session
    #sess = rt.InferenceSession(model_path)

    input_name = sess.get_inputs()[0].name
    loc_name = sess.get_outputs()[0].name
    conf_name = sess.get_outputs()[1].name
    mask_name = sess.get_outputs()[2].name
    priors_name = sess.get_outputs()[3].name
    proto_name = sess.get_outputs()[4].name
    priors = np.loadtxt(_prior_path, delimiter=',', dtype='float32')

    # print(os.listdir(path))
    # for idx, imgname in enumerate(os.listdir(path)):
    #     print(imgname)

    # for idx, imgname in enumerate(os.listdir(path)):
    #     print(idx, imgname)

    #imgpathname = path + "/" + imgname
    img = cv2.imread(imgname)

    img = cv2.resize(img, (275, 275))
    frame = torch.from_numpy(img).float()

    start = time.time()
    print(frame.unsqueeze(0).shape)
    batch = preprocess(frame.unsqueeze(0))
    print(time.time() - start)
    pred_onx = sess.run([loc_name, conf_name, mask_name, priors_name, proto_name], {input_name: batch.numpy()})

    predictions = {'loc': torch.from_numpy(pred_onx[0]), 'conf': torch.from_numpy(pred_onx[1]),
                   'mask': torch.from_numpy(pred_onx[2]), 'priors': torch.from_numpy(priors),
                   'proto': torch.from_numpy(pred_onx[4])}
    s1 = time.time()
    preds = detectonnx(predictions, _num_classes, bkg_label=_bkg_label, top_k=_top_k, conf_thresh=_conf_thresh,
                       nms_thresh=_nms_thresh)


    print("detectonnx", time.time() - s1)
    #img_numpy=preds
    img_numpy = prep_display(preds, frame, None, None, undo_transform=_undo_transform)
    print("total", time.time() - start)

    img_numpy = img_numpy[:, :, (2, 1, 0)]
    #print('img_numpy:',img_numpy)

    circle_ = detect_circle_by_hough(img_numpy)
    print('circle :',circle_)

    #
    # if len(circle_) > 0:
    #     cv2.circle(img_numpy, (circle_[0], circle_[1]), circle_[2], (0, 255, 0))
    if len(circle_) > 0:
        cv2.circle(img, (circle_[0], circle_[1]), circle_[2], (0, 255, 0), 4)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(output_dir + "/" + os.path.basename(imgname), img)



if __name__ == '__main__':
    model_path = "yolact_resnet50_275x275.onnx"
    sess = rt.InferenceSession(model_path)
    path="./image_1"
    for idx, imgname in enumerate(os.listdir(path)):
        evalimage(imgname=path + "/" + imgname,output_dir="./to_test_out",model_path=model_path,session=sess)

    '''
    path = './image_2'
    model_path = './yolox_s_512.onnx'
    starttime=time.time()
    session = loadmodel(model_path=model_path)

    print('load model',time.time()-starttime)

    for imgname in os.listdir(path):
        print('image name', imgname)
        starttime = time.time()
        for i in range(4):
            circledetection(input_shape=(512, 512), image_path=path + "/" + imgname, model='./yolox_s.onnx',
                            output_dir='./outputs', session=session)
        print('infer time', (time.time() - starttime)/4.0)
    
    '''
    print('end')