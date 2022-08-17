# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

import os, sys, csv
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset, PatchDataset, PatchIterd, GridPatchDataset, ShuffleBuffer, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss, GeneralizedDiceFocalLoss, GeneralizedDiceLoss, FocalLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet, UNet
from monai.networks.layers import Norm
from monai.optimizers import LearningRateFinder
from monai.transforms import (
    Activations,
    Activationsd,
    AddChannel,
    AddChanneld,
    AsDiscrete,
    AsDiscreted,
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    DivisiblePad,
    DivisiblePadd,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    Invertd,
    LabelFilter,
    LoadImaged,
    MapTransform,
    NormalizeIntensity,
    NormalizeIntensityd,
    Orientationd,
    Rand3DElasticd,
    Rand2DElasticd,
    RandAffine,
    RandAxisFlipd,
    RandBiasField,
    RandCropByPosNegLabeld,
    RandFlip,
    RandFlipd,
    RandRotated,
    RandScaleIntensity,
    RandScaleIntensityd,
    RandShiftIntensity,
    RandShiftIntensityd,
    RandSpatialCrop,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    RandZoomd,
    SaveImage,
    Spacing,
    Spacingd,
    SpatialCropd,
    SqueezeDimd,
)
from monai.utils import first, set_determinism


import torch

print_config()

# train_transforms = Compose(
#     [
#         # load 4 Nifti images and stack them together
#         LoadImaged(keys=["image", "label"]),
#         AddChanneld(keys=["image", "label"]),
#         CropForegroundd(keys=["image", "label"], source_key="image"),
#         # DivisiblePadd(keys=["image", "label"],k=32),
#         # RandAxisFlipd(keys=["image", "label"], prob=0.75),
#         RandRotated(keys=["image", "label"], prob=0.8, range_x=0.4, range_y=0.4, range_z=0.4, mode=["bilinear", "nearest"]),
#         RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.5, max_zoom=1.5, mode=["area", "nearest"]),
#         # Rand3DElasticd(keys=["image", "label"],
#         #     mode=("bilinear", "nearest"),
#         #     prob=1.0,
#         #     sigma_range=(5, 8),
#         #     magnitude_range=(100, 200),
#         #     translate_range=(50, 50, 2),
#         #     rotate_range=(np.pi / 36, np.pi / 36, np.pi),
#         #     scale_range=(0.15, 0.15, 0.15)
#         # ),
#         NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#         RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
#         RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
#         RandCropByPosNegLabeld(
#             keys=["image", "label"],
#             label_key="label",
#             image_key="image",
#             pos=7.0,
#             neg=1.0,
#             num_samples=4, 
#             spatial_size=[160, 160, 160]), # [96, 96, 96]
#         # CenterSpatialCropd(keys=["image", "label"], roi_size=(192, 224, 192)),
#         # DivisiblePadd(keys=["image", "label"],k=32),
#         EnsureTyped(keys=["image", "label"]),
#     ]
# )
train_transforms = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandRotated(keys=["image", "label"], prob=0.5, range_x=0.3, range_y=0.3, range_z=0.3, mode=["bilinear", "nearest"]),
        RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.5, max_zoom=1.5, mode=["area", "nearest"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.5, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.5, prob=1.0),
        RandCropByPosNegLabeld(
            keys=["image", "label"], 
            label_key="label",
            image_key="image",
            pos=7.0,
            neg=1.0,
            num_samples=2, 
            spatial_size=[96, 96, 96]), # [128, 128, 128]
        # CenterSpatialCropd(keys=["image", "label"], roi_size=(192, 224, 192)),
        # DivisiblePadd(keys=["image", "label"],k=32),
        EnsureTyped(keys=["image", "label"]),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        # DivisiblePadd(keys=["image", "label"],k=32),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ]
)

print("OK")

import glob

difficult = ["sub-r038s058", "sub-r050s003", "sub-r042s001", "sub-r039s002", "sub-r038s068", "sub-r038s026", 
             "sub-r038s007", "sub-r035s013", "sub-r019s001", "sub-r014s015", "sub-r014s010", "sub-r014s008", 
             "sub-r014s004", "sub-r014s002", "sub-r011s013", "sub-r011s011", "sub-r009s122", "sub-r009s113", 
             "sub-r009s106", "sub-r009s100", "sub-r009s082", "sub-r009s076", "sub-r009s074", "sub-r009s066", 
             "sub-r009s065", "sub-r009s062", "sub-r009s060", "sub-r009s058", "sub-r009s039", "sub-r009s036", 
             "sub-r009s035", "sub-r009s031", "sub-r009s028", "sub-r009s026", "sub-r009s016", "sub-r009s015", 
             "sub-r001s023", "sub-r001s022", "sub-r009s073", "sub-r009s012", "sub-r001s038", "sub-r009s014", 
             "sub-r009s013", "sub-r009s111", "sub-r009s056", "sub-r009s099", "sub-r009s061", "sub-r040s063", 
             "sub-r009s017", "sub-r049s016", "sub-r023s001", "sub-r049s011", "sub-r040s085", "sub-r009s050", 
             "sub-r038s097", "sub-r009s027", "sub-r038s021", "sub-r011s002", "sub-r010s014", "sub-r047s015", 
             "sub-r038s074", "sub-r009s094", "sub-r001s006", "sub-r003s010", "sub-r009s075", "sub-r010s010", 
             "sub-r009s090", "sub-r009s024", "sub-r009s030", "sub-r010s022", "sub-r005s055", "sub-r009s008", 
             "sub-r040s051"]

train_dir='/disk/febrian/Challenges_Data/ATLAS_2/data_ready/train_train/derivatives/ATLAS/'
train_images = glob.glob(os.path.join(train_dir, "**/*_T1w.nii.gz"), recursive=True)
train_labels = glob.glob(os.path.join(train_dir, "**/*_mask.nii.gz"), recursive=True)
train_names = [x.replace(train_dir, "").split("/")[0] for x in train_images]
print(f'There are {len(train_images)} subjects in our dataset.')
train_data_dicts = [
    {"image": image_name, "label": label_name, "name": name}
    for image_name, label_name, name in zip(train_images, train_labels, train_names)
]

print(f'There are {len(difficult)} difficult subjects in our dataset.')
train_images_difficult = [data for substring in difficult for data in train_images if substring in data]
train_labels_difficult = [data for substring in difficult for data in train_labels if substring in data]
print(f'There are {len(train_images_difficult)} difficult subjects in our dataset (images).')
print(f'There are {len(train_labels_difficult)} difficult subjects in our dataset (labels).')

train_data_dicts_difficult = [
    {"image": image_name, "label": label_name, "name": name}
    for image_name, label_name, name in zip(train_images_difficult, train_labels_difficult, difficult)
]

# Appending train data + diffciult data
for i in range(3):
    train_data_dicts = train_data_dicts + train_data_dicts_difficult
    print(f'There are {len(train_data_dicts)} difficult subjects in our dataset (total).')

print(f'There are {len(difficult)} difficult subjects in our dataset.')
train_images_difficult = [data for substring in difficult for data in train_images if substring in data]
train_labels_difficult = [data for substring in difficult for data in train_labels if substring in data]
print(f'There are {len(train_images_difficult)} difficult subjects in our dataset (images).')
print(f'There are {len(train_labels_difficult)} difficult subjects in our dataset (labels).')

train_ds = CacheDataset(
    data=train_data_dicts, transform=train_transforms,
    cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

val_dir='/disk/febrian/Challenges_Data/ATLAS_2/data_ready/train_val/derivatives/ATLAS/'
val_images = glob.glob(os.path.join(val_dir, "**/*_T1w.nii.gz"), recursive=True)
val_labels = glob.glob(os.path.join(val_dir, "**/*_mask.nii.gz"), recursive=True)
val_names = [x.replace(val_dir, "").split("/")[0] for x in train_images]
print(f'There are {len(val_images)} subjects in our dataset.')
val_data_dicts = [
    {"image": image_name, "label": label_name, "name": name}
    for image_name, label_name, name in zip(val_images, val_labels, val_names)
]
val_ds = CacheDataset(
    data=val_data_dicts, transform=val_transforms, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

val_interval = 1
VAL_AMP = True

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
# model = Probabilistic_Unet3D().to(device)
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16,32,64,128,256),    # (16,32,64)
    strides=(2, 2, 2, 2),           # (2, 2)
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

loss_function = GeneralizedDiceFocalLoss(
    to_onehot_y=False, 
    sigmoid=True, 
    softmax=False
    )

# loss_function = GeneralizedDiceLoss(
#     include_background=False,
#     to_onehot_y=False, 
#     sigmoid=True, 
#     softmax=False
#     )

# loss_function = FocalLoss(
#     include_background=False,
#     to_onehot_y=False,
#     gamma=2.0
#     )

# lower_lr, upper_lr = 1e-4, 1e-0
optimizer = torch.optim.Adam(model.parameters(), 0.00026366508987303583, weight_decay=1e-5)
# lr_finder = LearningRateFinder(model, optimizer, loss_function, device=device)
# lr_finder.range_test(train_loader, val_loader, end_lr=upper_lr, num_iter=20)
# steepest_lr, _ = lr_finder.get_steepest_gradient()
# print("steepest_lr: ", steepest_lr)

dice_metric = DiceMetric(include_background=False, reduction="mean")
dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")

# For one-hot and softmax
# post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
# post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

# for sigmoid
post_pred = Compose([
    EnsureType(), 
    Activations(sigmoid=True),
    AsDiscrete(threshold=0.5)])
post_label = Compose([EnsureType(), AsDiscrete()])

# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# Set deterministic training for reproducibility
set_determinism(seed=0)

print('READY')

best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
metric_values_tc = []

trained_model_dir='/disk/febrian/Challenges_Data/ATLAS_2/data_ready/'

# trained_model_name = 'unet_3D_v3_sigmoid_genDiceFocal_best_metric_model_epoch600'   # pre-trained u-net
# trained_mode_path = trained_model_dir + \
#     'models/' + trained_model_name + '.pth'
# model.load_state_dict(torch.load(
#     os.path.join(trained_mode_path)))

max_epochs = 50
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
best_model_path = "models/unet_3D_v3_patch_sigmoid_whole_genDiceFocalLoss_test_metric_model_epoch" + str(max_epochs)

total_start = time.time()
for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    steps_loss = 0
    
    epoch_len = len(train_ds) // train_loader.batch_size
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        
        # model.forward(inputs, labels, training=True)
        # elbo = model.elbo(labels)
        # reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + l2_regularisation(model.fcomb.layers)
        # loss = -elbo + 1e-5 * reg_loss
        
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        
        breaker = 10
        if step % (epoch_len / breaker) == 0:
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}"
                f", train_loss (avg/steps): {steps_loss / (epoch_len / breaker):.4f}"
                f", step time: {(time.time() - step_start):.4f}"
            )
            steps_loss = 0
            step_start = time.time()
        else:
            print(".", end="")
            steps_loss += loss.item()
    del batch_data, inputs, labels
            
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():

            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                # model.forward(val_inputs, val_labels, training=False)
                # val_outputs_o = model.sample(testing=True)
                                
                # define sliding window size and batch size for windows inference
                roi_size = (64, 64, 64)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model, overlap=0.25)

                # decollate the batch data into list of dictionaries, every dictionary maps to an input data                
                # val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                # val_labels = [post_trans(i) for i in decollate_batch(val_labels)]
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)
            del val_data, val_inputs, val_labels

            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            metric_values_tc.append(metric_tc)
            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(
                    model.state_dict(),
                    os.path.join(trained_model_dir, best_model_path + ".pth"),
                )
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
            
            # open the file in the write mode
            with open(os.path.join(best_model_path + ".csv"), 'a') as f:
                # create the csv writer
                writer = csv.writer(f)

                # write a row to the csv file
                writer.writerow([epoch_loss,metric])

    torch.save(
        model.state_dict(),
        os.path.join(trained_model_dir, best_model_path + "_per_epoch.pth"),
    )

    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start

torch.save(
    model.state_dict(),
    os.path.join(trained_model_dir, best_model_path + "_last.pth"),
)

plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.show()
plt.savefig(best_model_path + '.png')