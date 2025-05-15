import os
import sys

sys.path.append(os.getcwd())

import torch
import argparse
from omegaconf import OmegaConf
from cls.models.utils.node import LIFNode


neuron_outputs = []


def load_model(model_config):
    if model_config.name == "qkformer":
        from cls.models.static.qkformer_imagenet import qkformer_imagenet

        model = qkformer_imagenet()
    elif model_config.name == "spikformer":
        from cls.models.static.spikformer_imagenet import spikformer_imagenet

        model = spikformer_imagenet()
    else:
        raise ValueError("Invaild model name")

    return model


def record_output_lif(module, input, output):
    neuron_outputs.append(output)


def cal_firing_rate():
    all_slots = 0
    all_spikes = 0
    for t in neuron_outputs:
        all_slots += t.numel()
        all_spikes += t.sum()
    fr = all_spikes/all_slots
    return fr.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = config.ckpt_path

    model = load_model(config.model)
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"],
        strict=False,
    )
    model = model.to(device)
    model.eval()

    for name, submodule in model.named_modules():
        if isinstance(submodule, LIFNode):
            submodule.register_forward_hook(record_output_lif)

    dummy_data = torch.ones((1,3,224,224),device=device)
    model(dummy_data)
    print(cal_firing_rate())
