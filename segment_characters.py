# Necessary imports
import cv2
import imutils
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from skimage.filters import threshold_local

from read_character import KNNDigitClassifier
from read_data import *

#pytesseract.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/5.0.1/bin/tesseract'
PATH = "plates_dataset/"
annotations = os.listdir(PATH + 'annotations2')
images = os.listdir(PATH + 'images2')
images = sorted(images)


def show_plate(idx):
    print(images[idx])
    image = cv2.imread(PATH + 'images2/' + images[idx])
    data = read_xml()
    plates = data.loc[data["file"] == images[idx].split('.')[0]].reset_index()

    for i in range(plates.shape[0]):
        pts = np.array([[plates["xmin"][i], plates["ymin"][i]],
                        [plates["xmax"][i], plates["ymin"][i]],
                        [plates["xmax"][i], plates["ymax"][i]],
                        [plates["xmin"][i], plates["ymax"][i]]],
                       np.int32)

        plateBorder = cv2.polylines(image,
                                    [pts],
                                    True,
                                    (255, 0, 0),
                                    2)

    cv2.imshow("Image: " + images[idx], image)
    cv2.waitKey(1)
    print(images[idx])


def segmentate_plate(idx):
    data = read_xml()
    print(images[idx])
    plates = data.loc[data["file"] == images[idx].split('.')[0]].reset_index()
    image = cv2.imread(PATH + 'images2/' + images[idx])
    plate = image[plates["ymin"][0]:plates["ymax"][0],
            plates["xmin"][0]:plates["xmax"][0]]
    cv2.imshow("Segmentated plate" + images[idx], plate)
    cv2.waitKey(1)


def gray_blur_img(image, show=False):
    # sharpen + grayscale + blur
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    result = imutils.resize(image, width=600)
    result = cv2.filter2D(result, -1, kernel)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # result = cv2.GaussianBlur(result,(5,5),0)
    if show:
        cv2.imshow("Grayscale + blur", result)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    return result


def apply_threshold(image, show=False):
    # Inversed OTSU threshold
    mean = np.mean(image)
    binary = cv2.threshold(image, mean + (255 - mean) * 0.75, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if show:
        print(np.sum(binary == 255))
        print(np.sum(binary == 0))

    binary2 = cv2.bitwise_not(binary)

    if show:
        cv2.imshow("Threshold", binary)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        cv2.imshow("Threshold 2", binary2)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    return binary, binary2


def apply_threshold2(image, sharpen=False, show=False):
    result = image
    if sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        result = imutils.resize(image, width=600)
        result = cv2.filter2D(result, -1, kernel)
    V = cv2.split(cv2.cvtColor(result, cv2.COLOR_BGR2HSV))[2]
    T = threshold_local(V, 29, offset=15, method="gaussian")
    thresh = (V > T).astype("uint8") * 255
    thresh = cv2.bitwise_not(thresh)
    plate = imutils.resize(image, width=400)
    thresh = imutils.resize(thresh, width=400)

    thresh2 = cv2.bitwise_not(thresh)

    if show:
        cv2.imshow("Threshold", thresh)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        cv2.imshow("Threshold 2", thresh2)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    return thresh, thresh2


def kernel_operation(image, show=False):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    result = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel)
    result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_DILATE, kernel)
    # result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    # result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel3)

    if show:
        cv2.imshow("Kernel", result)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    return result


def sort_contours(contours):
    id = 0
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (cnts, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
                                        key=lambda b: b[1][id]))
    return cnts


def find_characters(plate, binary, show=False):
    cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    characters = []
    # standard width and height of characters (based on UK)
    d_width, d_height = 50, 80
    for c in sort_contours(cont):
        x, y, w, h = cv2.boundingRect(c)
        ratio = h / w
        if (0.75 <= ratio <= 3) and (h / plate.shape[0] >= 0.4):
            cv2.rectangle(plate, (x, y), (x + w, y + h), (0, 255, 0), 2)
            character = binary[y:y + h, x:x + w]
            character = cv2.resize(character, dsize=(2 * d_width, 2 * d_height))
            _, character = cv2.threshold(character,
                                         220,
                                         255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            character = cv2.copyMakeBorder(character,
                                           10, 10, 10, 10,
                                           cv2.BORDER_CONSTANT,
                                           value=0)
            characters.append(character)
    print("Detected {} letters...".format(len(characters)))

    if show and characters:
        draw_characters(characters)

    return characters


def draw_characters(characters):
    fig = plt.figure(figsize=(14, 4))
    grid = gridspec.GridSpec(ncols=len(characters),
                             nrows=1,
                             figure=fig)

    for i in range(len(characters)):
        fig.add_subplot(grid[i])
        plt.axis(False)
        plt.imshow(characters[i], cmap="gray")
    plt.show()


def segmentate_characters(idx, show=True):
    data = read_xml()
    print(images[idx])
    plates = data.loc[data["file"] == images[idx].split('.')[0]].reset_index()
    image = cv2.imread(PATH + 'images2/' + images[idx])
    plate = image[plates["ymin"][0]:plates["ymax"][0],
            plates["xmin"][0]:plates["xmax"][0]]
    new_plate = plate.copy()
    # new_plate = gray_blur_img(new_plate, show)
    # new_plate = apply_threshold(new_plate, show)
    new_plate, new_plate2 = apply_threshold2(new_plate, False, show)

    chars_nokernel = find_characters(plate, new_plate, show)
    chars_nokernel2 = find_characters(plate, new_plate2, show)

    new_plate = kernel_operation(new_plate, show)
    new_plate2 = kernel_operation(new_plate, show)
    characters = find_characters(plate, new_plate, show)
    characters2 = find_characters(plate, new_plate2, show)

    # Just for testing
    # Read the number plate
    text = pytesseract.image_to_string(plate, config='--psm 11')
    print("Detected license plate Number is:", text)

    if 3 <= len(characters) <= 20:
        return characters, 0, 0, 0

    if 3 <= len(characters2) <= 20:
        return characters2, 0, 0, 0

    return characters, characters2, chars_nokernel, chars_nokernel2


def main():
    idx = 1
    characters, characters2, chars_nokernel, chars_nokernel2 = segmentate_characters(idx,

                                                                                     True)
    classifier = KNNDigitClassifier()
    #classifier.train_preprocess()
    #classifier.train()
    classifier.load_model()
    print(str([c[0] for c in classifier.predict(characters)]))


if __name__ == "__main__":
    main()
