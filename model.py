import torch
import torch.nn as nn
import torchvision.models as models

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class EfficientNetV2TabRFusionWithSelfAttention(nn.Module):
    def __init__(self, tabr_input_dim, tabr_hidden_dim, num_classes, num_candidates, context_size):
        super(EfficientNetV2TabRFusionWithSelfAttention, self).__init__()
        
        # EfficientNetV2 part
        self.efficientnet = models.efficientnet_v2_s(weights='DEFAULT')
        efficientnet_out_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()  # Remove the final classification layer
        
        # TabR part
        self.tabr = TabR(tabr_input_dim, tabr_hidden_dim, tabr_hidden_dim, num_candidates, context_size)
        
        # Fusion
        self.fusion_dim = efficientnet_out_features + tabr_hidden_dim
        
        # Self-Attention
        self.self_attention = SelfAttention(embed_size=self.fusion_dim, heads=8)
        
        # Final classification
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, image, tabular_data, candidate_x, candidate_labels):
        # EfficientNetV2 forward pass
        img_features = self.efficientnet(image)
        
        # TabR forward pass
        tabr_features = self.tabr(tabular_data, candidate_x, candidate_labels)
        
        # Concatenate features
        combined_features = torch.cat((img_features, tabr_features), dim=1)
        
        # Reshape for self-attention (add sequence dimension)
        combined_features = combined_features.unsqueeze(1)
        
        # Apply self-attention
        attended_features = self.self_attention(combined_features, combined_features, combined_features)
        
        # Remove sequence dimension
        attended_features = attended_features.squeeze(1)
        
        # Final classification
        output = self.classifier(attended_features)
        
        return output

# TabR implementation (same as before)
class TabR(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_candidates, context_size):
        super(TabR, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_candidates = num_candidates
        self.context_size = context_size

        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Retrieval module
        self.key_transform = nn.Linear(hidden_dim, hidden_dim)
        self.value_transform = nn.Linear(hidden_dim, hidden_dim)
        self.label_embedding = nn.Embedding(output_dim, hidden_dim)
        
        # T module in value function
        self.T = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )

    def forward(self, x, candidate_x, candidate_labels):
        # Encode input and candidates
        x_encoded = self.encoder(x)
        candidates_encoded = self.encoder(candidate_x)

        # Compute keys
        x_key = self.key_transform(x_encoded)
        candidate_keys = self.key_transform(candidates_encoded)

        # Compute similarities
        similarities = -torch.cdist(x_key.unsqueeze(1), candidate_keys, p=2).squeeze(1)

        # Get top-k context
        _, top_k_indices = similarities.topk(self.context_size, dim=1)
        
        # Compute values
        candidate_values = self.value_transform(candidates_encoded)
        label_embeddings = self.label_embedding(candidate_labels)
        
        context_keys = torch.gather(candidate_keys, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        context_values = torch.gather(candidate_values, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        context_labels = torch.gather(label_embeddings, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        
        # Compute attention weights
        attention_weights = F.softmax(similarities, dim=1)
        
        # Compute context vector
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * (context_labels + self.T(x_key.unsqueeze(1) - context_keys)), dim=1)
        
        # Combine with input encoding
        combined = x_encoded + context_vector
        
        return combined

# Example usage
tabr_input_dim = 10
tabr_hidden_dim = 64
num_classes = 10
num_candidates = 1000
context_size = 96
image_size = 224

model = EfficientNetV2TabRFusionWithSelfAttention(tabr_input_dim, tabr_hidden_dim, num_classes, num_candidates, context_size)

# Dummy inputs
batch_size = 32
image = torch.randn(batch_size, 3, image_size, image_size)
tabular_data = torch.randn(batch_size, tabr_input_dim)
candidate_x = torch.randn(num_candidates, tabr_input_dim)
candidate_labels = torch.randint(0, num_classes, (num_candidates,))

output = model(image, tabular_data, candidate_x, candidate_labels)
print(output.shape)  # Should be (32, 10)
