import math

import cv2
import imutils  # image processing
import numpy as np

net = cv2.dnn.readNet('frozen_east_text_detection.pb')

outputLayers = []
outputLayers.append("feature_fusion/Conv_7/Sigmoid")
outputLayers.append("feature_fusion/concat_3")


def decodeBoundingBoxes(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[
        2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[
        3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if (score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                       offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0], sinA * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]


def join_boxes(boxes):
    return_boxes = []
    not_use_idx = []

    if (len(boxes) == 1):
        kx = np.array(cv2.boxPoints(boxes[0]))

        k_min = kx.min(axis=0)
        k_max = kx.max(axis=0)

        tl1 = k_min
        br1 = k_max
        return [[tl1, br1]]

    for ii, b in enumerate(boxes[:-1]):
        if ii not in not_use_idx:
            bx = np.array(cv2.boxPoints(b))

            b_min = bx.min(axis=0)
            b_max = bx.max(axis=0)

            tl1 = b_min  # [ b[0][0], b[0][1] ]
            br1 = b_max  # [ b[1][0], b[1][1] ]

            for jj, k in enumerate(boxes[ii + 1:], 1):
                if jj + ii not in not_use_idx:
                    kx = np.array(cv2.boxPoints(k))

                    k_min = kx.min(axis=0)
                    k_max = kx.max(axis=0)

                    tl2 = k_min  # [ k[0][0], k[0][1] ]
                    br2 = k_max  # [ k[1][0], k[1][1] ]

                    offset = 10

                    if ((tl1[0] > br2[0] or tl1[1] > br2[1] or
                         tl2[0] > br1[0] or tl2[1] > br1[1]) and
                            ~(tl1[1] + offset > tl2[1] and tl1[1] - offset < tl2[1] and
                              br1[1] + offset > br2[1] and br1[1] - offset < br2[
                                  1] and np.abs(tl1[0] - br2[0]) < 30) and
                            ~(tl1[1] + offset > tl2[1] and tl1[1] - offset < tl2[1] and
                              br1[1] + offset > br2[1] and br1[1] - offset < br2[
                                  1] and np.abs(tl2[0] - br1[0]) < 30) and
                            ~(tl2[1] + offset > tl1[1] and tl2[1] - offset < tl1[1] and
                              br2[1] + offset > br1[1] and br2[1] - offset < br1[
                                  1] and np.abs(tl1[0] - br2[0]) < 30) and
                            ~(tl2[1] + offset > tl1[1] and tl2[1] - offset < tl1[1] and
                              br2[1] + offset > br1[1] and br2[1] - offset < br1[
                                  1] and np.abs(tl2[0] - br1[0]) < 30)):
                        pass
                    else:
                        not_use_idx.append(ii + jj)
                        if ii not in not_use_idx:
                            not_use_idx.append(ii)

                        tl1 = [np.min([tl1[0], tl2[0]]),
                               np.min([tl1[1], tl2[1]])]

                        br1 = [np.max([br1[0], br2[0]]),
                               np.max([br1[1], br2[1]])]

            return_boxes.append([tl1, br1])

    if (len(boxes) - 1) not in not_use_idx:
        kx = np.array(cv2.boxPoints(boxes[-1]))

        k_min = kx.min(axis=0)
        k_max = kx.max(axis=0)

        tl1 = k_min
        br1 = k_max

        return_boxes.append([tl1, br1])

    return return_boxes


def join_boxes_overlaping(boxes):
    return_boxes = []
    not_use_idx = []

    if (len(boxes) == 1):
        k = boxes[0]
        return [k]

    for ii, b in enumerate(boxes[:-1]):
        if ii not in not_use_idx:
            tl1 = b[:2]
            br1 = b[2:]

            for jj, k in enumerate(boxes[ii + 1:], 1):
                if jj + ii not in not_use_idx:
                    tl2 = k[:2]
                    br2 = k[2:]

                    if ((tl1[0] > br2[0] or tl1[1] > br2[1] or
                         tl2[0] > br1[0] or tl2[1] > br1[1])):
                        pass
                    else:
                        not_use_idx.append(ii + jj)
                        if ii not in not_use_idx:
                            not_use_idx.append(ii)

                        tl1 = [np.min([tl1[0], tl2[0]]),
                               np.min([tl1[1], tl2[1]])]

                        br1 = [np.max([br1[0], br2[0]]),
                               np.max([br1[1], br2[1]])]

            return_boxes.append(tl1 + br1)

    if (len(boxes) - 1) not in not_use_idx:
        k = boxes[-1]
        return_boxes.append(k)

    return return_boxes


def overlaping(b1, b2):
    tl1 = b1[:2]
    br1 = b1[2:]

    tl2 = b2[:2]
    br2 = b2[2:]

    if (tl1[0] > br2[0] or tl1[1] > br2[1] or
            tl2[0] > br1[0] or tl2[1] > br1[1]):
        return False

    return True


def join_overlaping(b1, b2):
    tl1 = b1[:2]
    br1 = b1[2:]

    tl2 = b2[:2]
    br2 = b2[2:]

    tl = [np.min([tl1[0], tl2[0]]), np.min([tl1[1], tl2[1]])]

    br = [np.max([br1[0], br2[0]]), np.max([br1[1], br2[1]])]

    return tl + br


def detect_plate_east(im):
    resized = cv2.resize(im, (320, 320), interpolation=cv2.INTER_AREA)

    blob = cv2.dnn.blobFromImage(resized, 1.0, (320, 320), (123.68, 116.78, 103.94), True,
                                 False)

    net.setInput(blob)
    output = net.forward(outputLayers)
    scores = output[0]
    geometry = output[1]

    confThreshold = 0.5
    [boxes, confidences] = decodeBoundingBoxes(scores, geometry, confThreshold)

    nmsThreshold = 0.3
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)

    height_ = im.shape[0]
    width_ = im.shape[1]
    rW = width_ / float(320)
    rH = height_ / float(320)

    final_boxes = []
    for i in indices:
        # get 4 corners of the rotated rect
        final_boxes.append(boxes[i])

    boxes_out = []

    if (len(final_boxes) > 0):
        final_boxes = join_boxes(final_boxes)

        for b in final_boxes:

            x = b[0][0] * rW
            y = b[0][1] * rH

            xx = b[1][0] * rW
            yy = b[1][1] * rH

            if (xx - x) > (yy - y):
                im = cv2.rectangle(im, (int(x), int(y)), (int(xx), int(yy)), (255, 0, 0),
                                   thickness=2)

                boxes_out.append([int(x), int(y), int(xx), int(yy)])

    return im, boxes_out


im_comp = cv2.imread('plates_dataset/comp.jpeg')

gray_comp = cv2.cvtColor(im_comp.copy(), cv2.COLOR_BGR2GRAY)
roi_comp = \
cv2.threshold(gray_comp.copy(), 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def detect_plate(im, bx, bl=0):
    im_org = im.copy()

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert to grey scale

    if bl == 0:
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
    elif bl == 1:
        # better for big plates
        blur = cv2.bilateralFilter(gray, 13, 15, 15)
    elif bl == 2:
        # better fo small paltes
        blur = cv2.bilateralFilter(gray, 9, 75, 75)

    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, rectKern)

    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")

    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
    thresh_white = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = cv2.bitwise_and(thresh_white, thresh_white, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    final_boxes = []
    contours_boxes = []

    for bi, b1 in enumerate(bx):
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            b2 = [x, y, x + w, y + h]

            if bi == 0:
                contours_boxes.append(b2)

            if overlaping(b1, b2) and w > h:
                final_boxes.append(join_overlaping(b1, b2))

    if len(final_boxes) == 0:
        final_boxes = bx + contours_boxes

    if len(final_boxes) > 0:
        final_boxes = join_boxes_overlaping(final_boxes)

    out_plates = []
    out = []

    # loop over the license plate candidate contours    
    for b in final_boxes:

        x = int(b[0] if b[0] > 0 else 0)
        y = int(b[1] if b[1] > 0 else 0)

        xx = int(b[2] if b[2] > 0 else 0)
        yy = int(b[3] if b[3] > 0 else 0)

        im_w, im_h, _ = im.shape

        if (xx - x) > 1.5 * (yy - y) and (xx - x) * (yy - y) < im_w * im_h / 3:
            im = cv2.rectangle(im, (x, y), (xx, yy), (255, 0, 0), thickness=2)

            wt, ht = 0, 0  # int((xx-x)*0.1), int((yy-y)*0.2)

            licensePlate = im_org[y - ht:yy + ht, x - wt:xx + wt]

            gray = cv2.cvtColor(licensePlate.copy(), cv2.COLOR_BGR2GRAY)

            gray = cv2.resize(gray, (int(xx - x) * 3, int(yy - y) * 3))

            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])

            gray = cv2.filter2D(gray, -1, kernel)

            roi = \
            cv2.threshold(gray.copy(), 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[
                1]

            kernel = np.ones((2, 2), np.uint8)
            roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, )

            # roi = clear_border(roi)

            roi[roi > 0] = 1
            roi -= 1
            roi = np.abs(roi)

            r = 0.7 / 1

            h, w = roi.shape
            # left limit
            for i in range(w):
                if np.sum(roi[:, i]) > w * r:
                    break
            # right limit
            for j in range(w - 1, 0, -1):
                if np.sum(roi[:, j]) > w * r:
                    break

            for ii in range(h):
                if np.sum(roi[ii, :]) > h * r:
                    break
            for jj in range(h - 1, 0, -1):
                if np.sum(roi[jj, :]) > h * r:
                    break

            cropped = roi[ii:jj + 1, i:j + 1].copy()

            out_plates.append(cropped)
            out.append(licensePlate[ii // 3:jj // 3 + 1, i // 3:j // 3 + 1])

    if len(out_plates) > 0:
        compares_list = []
        for o in out:  # out_plates:
            # resized = cv2.resize(o, (roi_comp.shape[1], roi_comp.shape[0]), interpolation = cv2.INTER_AREA)
            # s = ssim(resized, roi_comp)

            h_comp = cv2.calcHist([gray_comp], [0], None, [256], [0, 256])

            gray_o = cv2.cvtColor(o.copy(), cv2.COLOR_BGR2GRAY)
            h_0 = cv2.calcHist([gray_o], [0], None, [256], [0, 256])

            c_comp = 0
            # Euclidean Distance between data1 and test
            i = 0
            while i < len(h_0) and i < len(h_comp):
                c_comp += (h_0[i] - h_comp[i]) ** 2
                i += 1
            c_comp = c_comp ** (1 / 2)

            compares_list.append(c_comp)

        min_idx = ([i[0] for i in sorted(enumerate(compares_list), key=lambda x: x[1])])[
            0]

        min_val = np.min(compares_list)

        out_plates = [out_plates[min_idx]]
    else:
        min_val = np.inf

    return out, im, out_plates, min_val


def return_plate(im):
    im1, bx = detect_plate_east(im.copy())
    plat_2, im2, _, v2 = detect_plate(im.copy(), bx)
    plat_3, im3, _, v3 = detect_plate(im.copy(), bx, 1)
    plat_4, im4, _, v4 = detect_plate(im.copy(), bx, 2)

    # im = cv2.hconcat([im1, im2])

    if v2 < v3 and v2 < v4:
        return plat_2
    elif v3 < v3 and v3 < v4:
        return plat_3
    else:
        return plat_4


image = cv2.imread('plates_dataset/' + 'images2/Cars0.png')
result = return_plate(image)
cv2.imshow("Resulting image", *result)
cv2.waitKey(1)
