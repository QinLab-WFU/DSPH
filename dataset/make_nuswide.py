import os
import re

import scipy.io as scio
import numpy as np

# mkdir mat
# mv make_nuswide.py mat
# python make_nuswide.py
root_dir = "/home/admin00/dataset/NUS-WIDE"


imageListFile = os.path.join(root_dir, "ImageList", "Imagelist.txt")
labelPath = os.path.join(root_dir, "Groundtruth", "AllLabels")
textFile = os.path.join(root_dir, "NUS_WID_Tags", "All_Tags.txt")
classIndexFile = os.path.join(root_dir, "ConceptsList", "Concepts81.txt")

# you can use the image urls to download images
imagePath = os.path.join("/home/admin00/dataset/nuswide/Flickr")

with open(imageListFile, "r") as f:
    indexs = f.readlines()

indexs = [os.path.join(imagePath, item.strip().replace("\\", "/")) for item in indexs]
print("indexs length:", len(indexs))

#class_index = {}
#with open(classIndexFile, "r") as f:
#    data = f.readlines()
#
#for i, item in enumerate(data):
#    class_index.update({item.strip(): i})

captions = []
with open(textFile, "r", encoding='utf-8') as f:
    for line in f:
        if len(line.strip()) == 0:
            print("some line empty!")
            continue
        caption = line.split()[1:]
        caption = " ".join(caption).strip()
        # caption = re.sub(r'[^a-zA-Z]+', "", str(caption))
        if len(caption) == 0:
             caption = "123456"
        captions.append(caption)

print("captions length:", len(captions))

#labels = np.zeros([len(indexs), len(class_index)], dtype=np.int8)
# label_lists = os.listdir(labelPath)
with open("/home/admin00/dataset/NUS-WIDE/Groundtruth/used_label.txt", encoding='utf-8') as f:
    label_lists = f.readlines()
label_lists = [item.strip() for item in label_lists]

class_index = {}
for i, item in enumerate(label_lists):
    class_index.update({item: i})

labels = np.zeros([len(indexs), len(class_index)], dtype=np.int8)

for item in label_lists:
    path = os.path.join(labelPath, item)
    class_label = item# .split(".")[0].split("_")[-1]

    with open(path, "r") as f:
        data = f.readlines()
    for i, val in enumerate(data):
        labels[i][class_index[class_label]] = 1 if val.strip() == "1" else 0
print("labels sum:", labels.sum())

not_used_id = []
with open("/home/admin00/dataset/NUS-WIDE/Groundtruth/not_used_id.txt", encoding='utf-8') as f:
    not_used_id = f.readlines()
not_used_id = [int(int(item.strip())-2) for item in not_used_id]

# for item in not_used_id:
#     indexs.pop(item)
#     captions.pop(item)
#     labels = np.delete(labels, item, 0)
ind = list(range(len(indexs)))
for item in not_used_id:
    ind.remove(item)
    indexs[item] = ""
    captions[item] = ""
indexs = [item for item in indexs if item != ""]
captions = [item for item in captions if item != ""]
ind = np.asarray(ind)
labels = labels[ind]
# ind = range(len(indexs))

print("indexs length:", len(indexs))
print("captions length:", len(captions))
print("labels shape:", labels.shape)

indexs = {"index": indexs}
captions = {"caption": captions}
labels = {"category": labels}

scio.savemat('/home/admin00/DSPH-main/dataset/nuswide/index.mat', indexs)
# scio.savemat("caption.mat", captions)
scio.savemat('/home/admin00/DSPH-main/dataset/nuswide/label.mat', labels)


captions = [item + "\n" for item in captions["caption"]]

with open('/home/admin00/DSPH-main/dataset/nuswide/caption.txt', "w", encoding='utf-8') as f:
    f.writelines(captions)

print("finished!")

