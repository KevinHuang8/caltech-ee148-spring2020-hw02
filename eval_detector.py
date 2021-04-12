import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    i0 = max(box_1[0], box_2[0])
    j0 = max(box_1[1], box_2[1])
    i1 = min(box_1[2], box_2[2])
    j1 = min(box_1[3], box_2[3])

    intersection_area = (i1 - i0) * (j1 - j0)
    if intersection_area <= 0:
        return 0
    if (i1 - i0) < 0 or (j1 - j0) < 0:
        return 0

    area1 = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    area2 = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])

    iou = intersection_area / (area1 + area2 - intersection_area)

    # print('---')
    # print(i0, j0, i1, j1)
    # print(iou, intersection_area, box_1, box_2, area1, area2)
    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        associated = []
        for i in range(len(gt)):
            closest_iou = -1
            closest = -1
            for j in range(len(pred)):
                if j in associated:
                    continue

                if pred[j][4] < conf_thr:
                    continue

                iou = compute_iou(pred[j][:4], gt[i])

                if iou < iou_thr:
                    continue

                if iou > closest_iou:
                    closest = j
                    closest_iou = iou

            if closest == -1:
                FN += 1
            else:
                associated.append(closest)
                TP += 1
    
        for j in range(len(pred)):
            if pred[j][4] < conf_thr:
                continue

            if j not in associated:
                FP += 1

    return TP, FP, FN

def plot_PR_curve(TP, FP, FN, **kwargs):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    plt.plot(recall, precision, **kwargs)

def gen_data(iou_thr, preds, gts):
    '''
    For a fixed IoU threshold, vary the confidence thresholds.
    Return TP, FP, and FN arrays for each confidence threshold.
    '''
    c = []
    for fname in preds:
        for box in preds[fname]:
            c.append(box[4])

    confidence_thrs = np.sort(np.array(c,dtype=float))
    tp = np.zeros(len(confidence_thrs))
    fp = np.zeros(len(confidence_thrs))
    fn = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp[i], fp[i], fn[i] = compute_counts(preds, gts, iou_thr=iou_thr, 
            conf_thr=conf_thr)

    return tp, fp, fn

def gen_plots(preds, gts, plot_title='PR Curves'):
    '''
    Run part g in the assignment for 'preds' and 'gts'.  
    '''
    iou_threshes = [0.25, 0.5, 0.75]

    for iou_thresh in iou_threshes:
        TP, FP, FN = gen_data(iou_thresh, preds, gts)
        plot_PR_curve(TP, FP, FN, label=f'iou thresh = {iou_thresh}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(plot_title)
    plt.legend()

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)

'''
Create PR Curves for training and test sets
'''

gen_plots(preds_train, gts_train, 'PR Curves for Training Set')

if done_tweaking:
    plt.figure()
    gen_plots(preds_test, gts_test, 'PR Curves for Test Set')

plt.show()
