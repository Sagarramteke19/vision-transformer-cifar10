import timm

def create_model(model_name='vit_tiny_patch16_224', num_classes=10, pretrained=False):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model
