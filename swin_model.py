import torch
import torch.nn as nn
import timm

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * nn.Sigmoid()(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = nn.Sigmoid()(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)
        

class CustomISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None):
        super(CustomISICModel, self).__init__()
        self.image_model = timm.create_model(model_name, pretrained=pretrained, 
                                             chekpoint_path=checkpoint_path)
        self.image_out_features = self.image_model.get_classifier().in_features
        self.image_model.reset_classifier(0)  # Remove the original classifier

        # Metadata part
        metadata_input_features = 6
        metadata_output_features = 128

        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_input_features, 128),
            nn.BatchNorm1d(128),
            Swish_Module(), #ReLU
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            Swish_Module(), #ReLU
            nn.Dropout(0.3),
            nn.Linear(256, metadata_output_features),
            nn.BatchNorm1d(metadata_output_features),
            Swish_Module() #ReLU
        )

        # Combine features from image model and metadata
        combined_features = self.image_out_features + metadata_output_features
        self.final_fc = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.BatchNorm1d(512),
            Swish_Module(), #ReLU
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, image, metadata):
        image_features = self.image_model(image)
        metadata_features = self.metadata_fc(metadata)
        combined_features = torch.cat((image_features, metadata_features), dim=1)
        output = self.final_fc(combined_features)
        
        return output

