import torch
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import natsort
import glob
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from  detectron2.structures import Boxes
def onlykeep_specific_classes(outputs):
    cls = outputs['instances'].pred_classes
    scores = outputs["instances"].scores
    boxes = outputs['instances'].pred_boxes
    masks = outputs['instances'].pred_masks
    # index to keep whose class
    #value == "helmet":
    indx_to_keep_0 = ((cls<=7) & (cls!=1) & (cls!=6)).nonzero().flatten().tolist()
    indx_to_keep_1 = (cls == 0).nonzero().flatten().tolist()

    # only keeping index  corresponding arrays
    cls0 = torch.tensor(np.take(cls.cpu().numpy(), indx_to_keep_0))
    scores0 = torch.tensor(np.take(scores.cpu().numpy(), indx_to_keep_0))
    boxes0 = Boxes(torch.tensor(np.take(boxes.tensor.cpu().numpy(), indx_to_keep_0, axis=0)))
    masks0 = torch.tensor(np.take(masks.cpu().numpy(), indx_to_keep_0, axis=0))

    cls1 = torch.tensor(np.take(cls.cpu().numpy(), indx_to_keep_1))
    scores1 = torch.tensor(np.take(scores.cpu().numpy(), indx_to_keep_1))
    boxes1 = Boxes(torch.tensor(np.take(boxes.tensor.cpu().numpy(), indx_to_keep_1, axis=0)))
    masks1 = torch.tensor(np.take(masks.cpu().numpy(), indx_to_keep_1, axis=0))


    # create new instance obj and set its fields
    obj = detectron2.structures.Instances(image_size=(1208, 1920))
    obj.set('pred_classes', cls0)
    obj.set('scores', scores0)
    obj.set('pred_boxes',boxes0)
    obj.set('pred_masks',masks0)
    return obj

path1=natsort.natsorted(glob.glob("paired_A/*.png"),reverse=False)
path2=natsort.natsorted(glob.glob("paired_B/*.png"),reverse=False)


'''
path3=natsort.natsorted(glob.glob("pytorch-CycleGAN-and-pix2pix/weather3/testA/*.png"),reverse=False)
path4=natsort.natsorted(glob.glob("pytorch-CycleGAN-and-pix2pix/weather3/testA/*.png"),reverse=False)
print(len(path4))
'''
for i in range(len(path1)):
    namex=path1[i]
    #namex="original/10-04-2021/"+"1004_"+path4[i].split("/")[-1]
    im = cv2.imread(namex)
    print(namex)
    name=path2[i]
    im2 = cv2.imread(name)
    print(name)
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    outputs2 = predictor(im2)

    obj=onlykeep_specific_classes(outputs)
    obj2=onlykeep_specific_classes(outputs2)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(obj.to("cpu"))
    v = Visualizer(im2[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out2 = v.draw_instance_predictions(obj2.to("cpu"))

    mask_array = obj.pred_masks.to("cpu").numpy()
    num_instances = mask_array.shape[0]
    scores = obj.scores.to("cpu").numpy()
    labels = obj.pred_classes .to("cpu").numpy()
    bbox   = obj.pred_boxes.to("cpu").tensor.numpy()

    mask_array = np.moveaxis(mask_array, 0, -1)

    mask_array_instance = []
    #img = np.zeros_like(im) #black
    h = im.shape[0]
    w = im.shape[1]
    img_mask = np.zeros([h, w, 3], np.uint8)
    img_mask.fill(255)
    color = (0, 0, 0)
    for j in range(num_instances):
        img = np.zeros_like(im)
        mask_array_instance.append(mask_array[:, :, j:(j+1)])
        img = np.where(mask_array_instance[j] == True, 255, img)
        array_img = np.asarray(img)
        im[np.where((array_img==[255,255,255]).all(axis=2))]=color
        im2[np.where((array_img==[255,255,255]).all(axis=2))]=color

        #img_mask = np.asarray(img_mask)
        #output = cv2.bitwise_and(im,im,mask = img_mask)
    mask_array2 = obj2.pred_masks.to("cpu").numpy()
    num_instances2 = mask_array2.shape[0]
    scores = obj2.scores.to("cpu").numpy()
    labels = obj2.pred_classes .to("cpu").numpy()
    bbox   = obj2.pred_boxes.to("cpu").tensor.numpy()

    mask_array2 = np.moveaxis(mask_array2, 0, -1)

    mask_array_instance2 = []
    #img = np.zeros_like(im) #black
    h2 = im2.shape[0]
    w2 = im2.shape[1]
    img_mask2 = np.zeros([h2, w2, 3], np.uint8)
    img_mask2.fill(255)
    color = (0, 0, 0)
    for j in range(num_instances2):
        img2= np.zeros_like(im)
        mask_array_instance2.append(mask_array2[:, :, j:(j+1)])
        img2 = np.where(mask_array_instance2[j] == True, 255, img2)
        array_img2 = np.asarray(img2)
        im2[np.where((array_img2==[255,255,255]).all(axis=2))]=color
        im[np.where((array_img2==[255,255,255]).all(axis=2))]=color
        #img_mask = np.asarray(img_mask)
        #output = cv2.bitwise_and(im,im,mask = img_mask)
    name1=namex.split("/")[-1]
    name2=name.split("/")[-1]

    #print(name1.split("_")[-1]).split("_")[]


    cv2.imwrite("tmp_data/A/"+name1,im)
    cv2.imwrite("tmp_data/B/"+name2,im2)
