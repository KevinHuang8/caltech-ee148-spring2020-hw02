import os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def load_image(filename):
    img = Image.open(filename)
    img.load()
    return img

def normalize_image(image):
    '''
    z-score normalize, sample-wise
    '''
    image_norm = (image - image.mean()) / image.std()
    return image_norm

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    p, q, _ = T.shape
    n, m, _ = I.shape

    result = np.zeros((n - p + 1, m - q + 1))

    for i in range(n - p + 1):
        for j in range(m - q + 1):
            section = I[i:i+p, j:j+q, :1]
            dot_product = np.sum(section*T)
            result[i, j] = dot_product

    return result

def predict_boxes(heatmap, kernel, quantile):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''
    output = []
    max_value = np.sum(kernel * kernel)

    threshold = np.quantile(heatmap, quantile)

    clustered = np.copy(heatmap)
    clustered[heatmap < threshold] = 0
    clustered[heatmap >= threshold] = 255

    n_clusters = flood_fill(clustered, limit=kernel.shape[0]*kernel.shape[1])
    centers = get_centers(clustered, n_clusters)
    confidences = get_confidence(heatmap, clustered, max_value, n_clusters)

    for ind, (i, j) in enumerate(centers):
        output.append([i, j, i + kernel.shape[0], j + kernel.shape[1],
            confidences[ind]])

    return output

def flood_fill(arr, limit=20, ecc_thresh=1.5):
    '''
    Takes in an array (corresponding to the result array after applying the
    kernel) that has values of either 0 or 255, with 255 corresponding to pixels
    that have been identified as part of a traffic light. This function clusters
    all of the pixels by assigning a different integer to each pixel group.

    limit - maximum number of pixels allowed in a cluster. If this is exceeded,
    that cluster is discarded, as traffic lights should not be very big 
    (they should not exceed the size of the filter).
    '''
    n = 1
    while 255 in arr:
        loc = np.argwhere(arr == 255)[0]
        flood_fill_helper(arr, loc, n, set())
        count = len(np.argwhere(arr == n))
        if count > limit:
            arr[arr == n] = 0
        # if eccentricity(arr, n) > ecc_thresh:
        #     arr[arr == n] = 0
        n += 1
    return n - 1

def flood_fill_helper(arr, loc, cluster_num, visited):
    i, j = loc[0], loc[1]
    visited.add((i, j))
    if (arr[i, j] != 255):
        return
    arr[i, j] = cluster_num
    if (i + 1, j) not in visited and i < arr.shape[0] - 1:
        flood_fill_helper(arr, (i + 1, j), cluster_num, visited)
    if (i, j + 1) not in visited and j < arr.shape[1] - 1:
        flood_fill_helper(arr, (i, j + 1), cluster_num, visited)
    if (i - 1, j) not in visited and i > 0:
        flood_fill_helper(arr, (i - 1, j), cluster_num, visited)
    if (i, j - 1) not in visited and j > 0:
        flood_fill_helper(arr, (i, j - 1), cluster_num, visited)

def get_centers(arr, n_clusters):
    '''
    Returns a list of coordinates corresponding to the centers for each
    out of n_clusters clusters in arr.
    '''
    centers = []
    for i in range(1, n_clusters + 1):
        matches = np.transpose((arr == i).nonzero())
        if matches.size == 0:
            continue
        centers.append(np.mean(matches, axis=0))
    return centers

def get_confidence(heatmap, arr, max_value, n_clusters):
    '''
    Returns a list of confidence values for each of n_cluster clusters.
    arr is the clustered version of heatmap (after flood filling)
    max_value is the maximum value that the heatmap can have, corresponding
    to 100% confidence.
    '''

    confidences = []
    for i in range(1, n_clusters + 1):
        matches = (arr == i).nonzero()
        if np.transpose(matches).size == 0:
            continue

        val = np.max(heatmap[matches])
        s = 0.5*max_value
        c = np.exp(-((val - max_value)/s)**2)
        confidences.append(c)
    return confidences

def eccentricity(arr, n):
    '''
    Returns the "eccentricity" of cluster n in arr. Eccentricity is defined
    as the ratio of the range of the cluster along axis to the range along the
    other axis. Traffic light clusters are roughly circular, so high
    eccentricities should be discarded.
    '''
    matches = np.transpose((arr == n).nonzero())
    if matches.size == 0:
        return 1

    mins = np.min(matches, axis=0)
    maxes = np.max(matches, axis=0)
    
    dx = np.abs(mins[1] - maxes[1]) + 1
    dy = np.abs(mins[0] - maxes[0]) + 1

    larger = max(dx, dy)
    smaller = min(dx, dy)

    return larger / smaller

def stop_overlap(bounding_boxes):
    '''
    Takes a list of bounding boxes and for any overlapping ones, take the
    one with the maximum confidence.
    '''
    processed = set()

    valid = set()

    for i, (xmin1, ymin1, xmax1, ymax1, c1) in enumerate(bounding_boxes):
        if i in processed:
            continue
        processed.add(i)
        overlapping = [i]
        for j, (xmin2, ymin2, xmax2, ymax2, c2) in enumerate(bounding_boxes):
            if i >= j:
                continue

            if j in processed:
                continue

            if overlap_1D((xmin1, xmax1), (xmin2, xmax2)) and \
                overlap_1D((ymin1, ymax1), (ymin2, ymax2)):
                overlapping.append(j)
                processed.add(j)

        max_c = -1
        max_box = -1
        for k in overlapping:
            if bounding_boxes[k][4] > max_c:
                max_c = bounding_boxes[k][4]
                max_box = k
        valid.add(max_box)

    new_boxes = []
    for i, box in enumerate(bounding_boxes):
        if i in valid:
            new_boxes.append(box)
    return new_boxes

def overlap_1D(interval1, interval2):
    return (interval1[1] >= interval2[0] and interval2[1] >= interval1[0]) \
        or (interval2[1] >= interval1[0] and interval1[1] >= interval2[0])

def display_results(image, bounding_boxes):
    '''
    Draws the bounding boxes given in bounding_boxes on image, and then
    displays it.
    '''
    im = Image.fromarray(image.astype('uint8'), 'RGB')
    draw = ImageDraw.Draw(im)
    for i0, j0, i1, j1, c in bounding_boxes:
        draw.rectangle((j0, i0, j1, i1), outline='red')
        draw.text((j0, i0 - 10), f'{c: .4f}', fill=(255, 255, 255, 255))
    im.show()

def display_results2(image, bounding_boxes):
    '''
    Draws the bounding boxes given in bounding_boxes on image, and then
    displays it.
    '''
    im = Image.fromarray(image.astype('uint8'), 'RGB')
    draw = ImageDraw.Draw(im)
    for i0, j0, i1, j1 in bounding_boxes:
        draw.rectangle((j0, i0, j1, i1), outline='red')
    im.show()

def _detect_red_light(image, kernel, quantile, display=False):
    '''
    Helper function.
    '''    
    image_norm = normalize_image(image)
    heatmap = compute_convolution(image_norm, kernel)
    return predict_boxes(heatmap, kernel, quantile)

def detect_red_light_mf(I, display=False):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    # Paramters for the algorithm
    scaling_factors = [1, 1/2, 1/3]
    quantile = 0.9995
    kernel_name = 'filter2.png'
    kernel = load_image(kernel_name)

    output = []
    kernel = np.asarray(kernel, dtype='int32')[:,:,:3]
    kernel_shape = kernel.shape

    # Try different scale kernels, stopping if enough traffic lights
    # have been found
    for s in scaling_factors:
        kernel = load_image(kernel_name)
        if s != 1:
            kernel = kernel.resize((int(kernel_shape[1]*s), int(kernel_shape[0]*s)))
        kernel = np.asarray(kernel, dtype='int32')[:,:,:3]
        kernel = normalize_image(kernel)

        image_norm = normalize_image(I)
        heatmap = compute_convolution(image_norm, kernel)
        if display:
            plt.imshow(heatmap)
            plt.figure()
        bb = predict_boxes(heatmap, kernel, quantile)
        # if len(bb) >= attempts_per_scale:
        #     output.extend(bb)
        #     break

        output.extend(bb)

    output = stop_overlap(output)

    if display:
        plt.show()
        display_results(I, output)
    
    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''

preds_train = {}
for i in range(len(file_names_train)):
    print('TRAIN', i)

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I, display=False)

#save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):
        print('TEST', i)

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
