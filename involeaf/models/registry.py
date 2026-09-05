"""Single entry point for model construction, so configs stay declarative."""

from __future__ import annotations

import timm
import torch.nn as nn

from involeaf.models.agritl_vit import build_agritl_vit
from involeaf.models.vit_involution import build_involeaf, new_parameter_names

ARCHITECTURES = ("timm", "involeaf", "agritl_vit")


def build_model(cfg: dict, num_classes: int) -> nn.Module:
    """Build a model from a config dict.

    ``cfg['arch']`` selects the family:
      timm        -- an unmodified timm model (ViT-B/16 baseline, ResNet-50)
      involeaf    -- ViT with involution in the FFN slot
      agritl_vit  -- the AgriTL-ViT reproduction
    """
    arch = cfg.get("arch", "timm")
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHITECTURES}")

    backbone = cfg["backbone"]
    pretrained = cfg.get("pretrained", True)

    if arch == "timm":
        kwargs = {}
        # ResNet has no position embedding, so dynamic_img_size does not apply to it.
        if "vit" in backbone:
            kwargs["dynamic_img_size"] = cfg.get("dynamic_img_size", True)
        model = timm.create_model(
            backbone, pretrained=pretrained, num_classes=num_classes, **kwargs
        )
        model.involeaf_meta = {"backbone": backbone, "architecture": "baseline"}
        return model

    if arch == "agritl_vit":
        return build_agritl_vit(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            dual_attention=cfg.get("dual_attention", True),
            resnet_module=cfg.get("resnet_module", True),
            dropout=cfg.get("dropout", 0.1),
        )

    return build_involeaf(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        variant=cfg.get("variant", "inv_ffn"),
        inv_blocks=cfg.get("inv_blocks", "all"),
        kernel_size=cfg.get("kernel_size", 7),
        group_channels=cfg.get("group_channels", 16),
        reduction_ratio=cfg.get("reduction_ratio", 4),
        norm_layer=cfg.get("norm_layer", "gn"),
        impl=cfg.get("impl", "shift"),
        delta_init=cfg.get("delta_init", True),
        ffn_ratio=cfg.get("ffn_ratio", 2.0),
        ffn_init=cfg.get("ffn_init", "slice"),
        cls_mode=cfg.get("cls_mode", "identity"),
    )


def param_groups(model: nn.Module, cfg: dict) -> list[dict]:
    """Layer-wise learning rates.

    Modules introduced by surgery are randomly initialised inside a pretrained encoder
    and need a higher learning rate than the pretrained blocks; the classifier head is
    entirely new and needs higher still. Training everything at one rate either destroys
    the pretrained features or starves the new ones.
    """
    lr = float(cfg.get("lr", 1e-4))
    lr_backbone = float(cfg.get("lr_backbone", lr * 0.1))
    lr_head = float(cfg.get("lr_head", lr * 10))
    weight_decay = float(cfg.get("weight_decay", 0.05))

    new_names = new_parameter_names(model)
    head_prefixes = ("head.", "fc.", "classifier.")

    groups: dict[str, dict] = {
        "head": {"params": [], "lr": lr_head, "weight_decay": weight_decay},
        "new": {"params": [], "lr": lr, "weight_decay": weight_decay},
        "pretrained": {"params": [], "lr": lr_backbone, "weight_decay": weight_decay},
        "no_decay": {"params": [], "lr": lr_backbone, "weight_decay": 0.0},
    }

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(head_prefixes):
            groups["head"]["params"].append(p)
        elif name in new_names:
            groups["new"]["params"].append(p)
        elif p.ndim <= 1:            # norms, biases, cls token, position embedding
            groups["no_decay"]["params"].append(p)
        else:
            groups["pretrained"]["params"].append(p)

    for key, g in groups.items():
        g["name"] = key
    return [g for g in groups.values() if g["params"]]
