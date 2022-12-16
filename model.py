import torch
import torchvision
import time
import cv2
import numpy as np

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def clip_coords(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])  
        boxes[:, 1].clamp_(0, shape[0])  
        boxes[:, 2].clamp_(0, shape[1])  
        boxes[:, 3].clamp_(0, shape[0])
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])

def scale_coords(
    img1_shape,
    coords,
    img0_shape,
    ratio_pad = None
):
    if ratio_pad is None:  
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  
    coords[:, [1, 3]] -= pad[1]  
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def letterbox(
    im, 
    new_shape = (640, 640), 
    color = (114, 114, 114),
    auto = True,
    scaleFill = False,
    scaleup = True,
    stride = 32
):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  
    elif scaleFill:  
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  

    dw /= 2  
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation = cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value = color)

    return im, ratio, (dw, dh)

def box_iou(box1, box2):

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)

def non_max_suppression(
    prediction, 
    conf_thres = 0.25, 
    iou_thres = 0.45, 
    classes = None, 
    agnostic = False, 
    multi_label = False, 
    labels = (), 
    max_det = 300
):
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres

    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    min_wh, max_wh = 2, 4096
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh) 
        boxes, scores = x[:, :4] + c, x[:, 4] 
        i = torchvision.ops.nms(boxes, scores, iou_thres)  
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3): 
            iou = box_iou(boxes[i], boxes) > iou_thres  
            weights = iou * scores[None]  
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True) 
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break 

    return output


def inference(session, img0, imgsz = 1280, conf = 0.2, iou = 0.45):
    img = letterbox(img0, imgsz, stride = 64, auto = False)[0]
    img = img.transpose((2, 0, 1))[::-1] 
    img = np.ascontiguousarray(img)

    img = img.astype('float32')
    img = img / 255.0

    if len(img.shape) == 3:
        img = img[None]

    pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img})[0])
    pred = non_max_suppression(pred, conf, iou, None, False, max_det = 1000)

    im0 = img0.copy()
    det = pred[0]

    boxes = []
    labels = []

    if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # for c in det[:, -1].unique():
        #     n = (det[:, -1] == c).sum()

        for *xyxy, conf, cls in reversed(det):
            class_idx = int(cls)
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

            boxes.append((c1, c2))
            labels.append(class_idx)
    
    return boxes, labels

if __name__ == '__main__':
    import onnxruntime
    import os

    import time

    session = onnxruntime.InferenceSession(os.path.join('assets', 'relative.obj.ctx'), providers = ['CUDAExecutionProvider'])
    img = cv2.cvtColor(cv2.imread('test.png')[:, :, :3], 4)

    for _ in range(100):
        s = time.time()
        inference(session, img)
        print(time.time() - s)

    for box in inference(session, img):
        c1, c2 = box
        print(c1, c2)