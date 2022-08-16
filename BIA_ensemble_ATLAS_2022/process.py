import numpy as np
from settings import loader_settings, loader_settings_test
import medpy.io
import os, pathlib

## TEST
from monai.data import decollate_batch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    Activationsd,
    Compose,
    EnsureType,     
    EnsureTyped,
    AsDiscrete,
    NormalizeIntensity,
    NormalizeIntensityd,
)

import torch
import torch.nn.functional as F


def connected_components_3D(image: torch.Tensor, num_iterations: int = 75) -> torch.Tensor:
    r"""Computes the Connected-component labelling (CCL) algorithm.
    .. image:: https://github.com/kornia/data/raw/main/cells_segmented.png
    The implementation is an adaptation of the following repository:
    https://gist.github.com/efirdc/5d8bd66859e574c683a504a4690ae8bc
    .. warning::
        This is an experimental API subject to changes and optimization improvements.
    .. note::
    See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
    connected_components.html>`__.
    Args:
        image: the binarized input image with shape :math:`(*, 1, H, W)`.
        The image must be in floating point with range [0, 1].
        num_iterations: the number of iterations to make the algorithm to converge.
    Return:
        The labels image with the same shape of the input image.
    Example:
        >>> img = torch.rand(2, 1, 4, 5)
        >>> img_labels = connected_components(img, num_iterations=100)
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input imagetype is not a torch.Tensor. Got: {type(image)}")

    if not isinstance(num_iterations, int) or num_iterations < 1:
        raise TypeError("Input num_iterations must be a positive integer.")

    if len(image.shape) < 4 or image.shape[-4] != 1:
        raise ValueError(f"Input image shape must be (*,1,H,W,Z). Got: {image.shape}")

    H, W, Z = image.shape[-3:]
    image_view = image.view(-1, 1, H, W, Z)

    # precompute a mask with the valid values
    mask = image_view == 1

    # allocate the output tensors for labels
    B, _, _, _, _ = image_view.shape
    out = torch.arange(B * H * W * Z, device=image.device, dtype=image.dtype).view((-1, 1, H, W, Z))
    out[~mask] = 0

    for _ in range(num_iterations):
        out[mask] = F.max_pool3d(out, kernel_size=3, stride=1, padding=1)[mask]

    return out.view_as(image)

def post_processing(temp, min_blob_size):
    # POST-PORCESSING
    cc_output = connected_components_3D(
        temp, 
        num_iterations=200
        )
    blob_size = []
    for ula in torch.unique(cc_output):
        if ula != 0:
            the_label = torch.count_nonzero((cc_output == ula))
            blob_size.append(the_label.item())
            if the_label < min_blob_size:
                temp[cc_output == ula] = 0
    
    return temp

class Seg():
    def __init__(self):
        # super().__init__(
        #     validators=dict(
        #         input_image=(
        #             UniqueImagesValidator(),
        #             UniquePathIndicesValidator(),
        #         )
        #     ),
        # )
        return
            
    def process(self):
        debug = False

        ## LOAD ALL MODELS
        view = 224
        overlap = 0
        min_blob_size = 27

        path = os.getcwd() + "/"

        trained_model_dir = "trained_models"
        post_pred_softmax = Compose([EnsureType(), Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=2)])
        post_pred_025 = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.25)])

        val_transforms = Compose(
            [
                # LoadImaged(keys=["image", "label"]),
                # AddChanneld(keys=["image", "label"]),
                NormalizeIntensity(nonzero=True, channel_wise=True),
                EnsureType(),
            ]
        )

        device = torch.device("cuda:0")
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16,32,64,128,256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)
        trained_model_name = '/unet_3D_v3_softmax_genDiceFocal_best_metric_model_epoch600'
        trained_mode_path = trained_model_dir + trained_model_name + '.pth'
        model.load_state_dict(torch.load(
            os.path.join(trained_mode_path)))
        model.eval()

        model_hardData = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16,32,64,128,256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)
        trained_model_name = '/unet_3D_patch96_softmax_genDiceFocalLoss_wHardDataOnly_best_preTrained_metric_model_epoch600'
        trained_mode_path = trained_model_dir + trained_model_name + '.pth'
        model_hardData.load_state_dict(torch.load(
            os.path.join(trained_mode_path)))
        model_hardData.eval()

        model_sigmoid = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16,32,64,128,256),    # (16,32,64)
            strides=(2, 2, 2, 2),           # (2, 2)
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)
        trained_model_name = '/unet_3D_v3_sigmoid_genDiceFocal_best_metric_model_epoch600'
        trained_mode_path = trained_model_dir + trained_model_name + '.pth'
        model_sigmoid.load_state_dict(torch.load(
            os.path.join(trained_mode_path)))
        model_sigmoid.eval()

        model_sigmoid_hardData = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16,32,64,128,256),    # (16,32,64)
            strides=(2, 2, 2, 2),           # (2, 2)
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)
        trained_model_name = '/unet_3D_patch96_sigmoid_genDiceFocalLoss_wHardDataOnly_best_preTrained_metric_model_epoch600'
        trained_mode_path = trained_model_dir + trained_model_name + '.pth'
        model_sigmoid_hardData.load_state_dict(torch.load(
            os.path.join(trained_mode_path)))
        model_sigmoid_hardData.eval()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if debug:
            inp_path = loader_settings_test['InputPath']  # Path for the input
            out_path = loader_settings_test['OutputPath']  # Path for the output
        else:
            inp_path = loader_settings['InputPath']  # Path for the input
            out_path = path + loader_settings['OutputPath']  # Path for the output
        file_list = os.listdir(inp_path)  # List of files in the input
        file_list = [os.path.join(inp_path, f) for f in file_list]
        for fil in file_list:
            dat, hdr = medpy.io.load(fil)  # dat is a numpy array
            im_shape = dat.shape
            dat = dat.reshape(1, 1, *im_shape)  # reshape to Pytorch standard
            # Convert 'dat' to Tensor, or as appropriate for your model.

            ###########
            ### Replace this section with the call to your code.
            val_inputs = val_transforms(dat).to(device)

            sw_batch_size = 32
            roi_size = (view, view, view)
            val_outputs_96whole = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model, overlap=overlap
            )
            val_outputs_96whole = post_pred_softmax(decollate_batch(val_outputs_96whole))
            val_outputs_96whole = torch.unsqueeze(torch.unsqueeze(val_outputs_96whole[0][1], 0), 0)
            val_outputs_96whole = post_processing(val_outputs_96whole, min_blob_size)    
            
            val_outputs_96whole_hardData = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model_hardData, overlap=overlap
            )
            val_outputs_96whole_hardData = post_pred_softmax(decollate_batch(val_outputs_96whole_hardData))
            val_outputs_96whole_hardData = torch.unsqueeze(torch.unsqueeze(val_outputs_96whole_hardData[0][1], 0), 0)
            val_outputs_96whole_hardData = post_processing(val_outputs_96whole_hardData, min_blob_size)
            
            val_outputs_96whole_sigmoid_raw = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model_sigmoid, overlap=overlap
            )
            val_outputs_96whole_sigmoid = post_pred_025(decollate_batch(val_outputs_96whole_sigmoid_raw))
            val_outputs_96whole_sigmoid = torch.unsqueeze(val_outputs_96whole_sigmoid[0], 0)
            val_outputs_96whole_sigmoid = post_processing(val_outputs_96whole_sigmoid, min_blob_size)
            
            val_outputs_96whole_sigmoid_raw = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model_sigmoid_hardData, overlap=overlap
            )
            val_outputs_96whole_sigmoid_hardData = post_pred_025(decollate_batch(val_outputs_96whole_sigmoid_raw))
            val_outputs_96whole_sigmoid_hardData = torch.unsqueeze(val_outputs_96whole_sigmoid_hardData[0], 0)
            val_outputs_96whole_sigmoid_hardData = post_processing(val_outputs_96whole_sigmoid_hardData, min_blob_size)
                            
            preds_list = [
                val_outputs_96whole, 
                val_outputs_96whole_hardData,
                val_outputs_96whole_sigmoid,
                val_outputs_96whole_sigmoid_hardData
            ]

            prediction_mean = torch.ge(torch.mean(torch.stack(preds_list),dim=0), 0.5 ).float()
            # prediction_mean = [post_processing(prediction_mean, min_blob_size)]
            dat = post_processing(prediction_mean, min_blob_size)

            ###
            ###########

            dat = dat.reshape(*im_shape).cpu()
            out_name = os.path.basename(fil)
            out_filepath = os.path.join(out_path, out_name)
            medpy.io.save(dat, out_filepath, hdr=hdr)

            # print("fil: ", fil)
            # print(f'=== saving [{out_filepath}] from [{fil}] ===')
            # print("val_inputs.shape : ", val_inputs.shape)
            # print("dat.shape        : ", dat.shape)


            # out_filepath = os.path.join(out_path, 'output.txt')
            # f = open(out_filepath, 'w')
            # f.write('Hello World')
            # f.close()
            # print("DONE")
        return


if __name__ == "__main__":
    pathlib.Path("/output/images/stroke-lesion-segmentation/").mkdir(parents=True, exist_ok=True)
    Seg().process()
