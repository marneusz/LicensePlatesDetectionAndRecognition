# Necessary imports
import glob
import os
import xml.etree.ElementTree as ET

import pandas as pd


class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="",
                  flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size

def read_xml():
    PATH = "plates_dataset/"
    annotations = os.listdir(PATH + 'annotations2')
    images = os.listdir(PATH + 'images2')

    # parsing xml files similar to https://www.kaggle.com/stpeteishii/car-plate-get-annotation-info-from-xml

    dataset = {
        "file": [],
        "width": [],
        "height": [],
        "xmin": [],
        "xmax": [],
        "ymin": [],
        "ymax": [],
    }

    for annotation in glob.glob(PATH + "annotations2/*.xml"):
        # representing xml files as a tree
        tree = ET.parse(annotation)

        for element in tree.iter():
            if 'size' in element.tag:
                for attribute in list(element):
                    if 'width' in attribute.tag:
                        width = int(round(float(attribute.text)))
                    if 'height' in attribute.tag:
                        height = int(round(float(attribute.text)))

            if 'object' in element.tag:
                for attribute in list(element):

                    if 'name' in attribute.tag:
                        dataset['width'].append(width)
                        dataset['height'].append(height)
                        dataset['file'].append(annotation.split('/')[-1].split('.')[0])

                    if 'bndbox' in attribute.tag:
                        for dim in list(attribute):
                            if 'xmin' in dim.tag:
                                xmin = int(round(float(dim.text)))
                                dataset['xmin'].append(xmin)
                            if 'ymin' in dim.tag:
                                ymin = int(round(float(dim.text)))
                                dataset['ymin'].append(ymin)
                            if 'xmax' in dim.tag:
                                xmax = int(round(float(dim.text)))
                                dataset['xmax'].append(xmax)
                            if 'ymax' in dim.tag:
                                ymax = int(round(float(dim.text)))
                                dataset['ymax'].append(ymax)
    data = pd.DataFrame(dataset)
    data = data.sort_values('file').reset_index(drop=True)
    return data
