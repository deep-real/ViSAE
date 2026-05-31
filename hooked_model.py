import torch
from transformers import AutoImageProcessor, Dinov2Model, CLIPVisionModel
from PIL import Image
from torchvision import transforms

class HookedModel:
    def __init__(self, model_name='facebook/dinov2-base', transcoder=False, hook_points=None, device='cuda:0', torch_dtype=torch.bfloat16):
        self.model_name = model_name
        if 'clip' in model_name:
            self.model = CLIPVisionModel.from_pretrained(model_name, torch_dtype=torch_dtype).vision_model.to(device)
        elif 'dinov2' in model_name:
            self.model = Dinov2Model.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
        self.model.eval()
        self.preprocessor = AutoImageProcessor.from_pretrained(model_name)
        self.device = device if device is not None else 'cpu'
        self.torch_dtype = torch_dtype
        self.hooks = []
        self.hook_points = hook_points
        self.transcoder = transcoder

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.activations = {}

        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        def hook_fn(hook_point):
            def hook(module, input_args, output_tensor):
                if self.transcoder:
                    self.activations[hook_point + '_out'] = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
                    self.activations[hook_point + '_in'] = input_args[0] if isinstance(input_args, tuple) else input_args
                else:
                    self.activations[hook_point] = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
                return output_tensor
            return hook
        
        # Remove any existing hooks
        self.remove_hooks()
        
        # Register new hooks
        for hook_point in self.hook_points:
            layer, module_name = hook_point.split('_')
            layer_num = int(layer.split('-')[-1])  # Get the layer index
            if 'dinov2' in self.model_name:
                if 'mlp' in module_name:
                    module = self.model.encoder.layer[layer_num].mlp
                elif 'resid' in module_name:
                    module = self.model.encoder.layer[layer_num]
            elif 'clip' in self.model_name:
                if 'mlp' in module_name:
                    module = self.model.encoder.layers[layer_num].mlp
                elif 'resid' in module_name:
                    module = self.model.encoder.layers[layer_num]
            hook = module.register_forward_hook(hook_fn(hook_point))
            self.hooks.append(hook)
    
    @torch.no_grad()
    def get_activations_from_image(self, image):
        self.activations.clear()
        inputs = self.preprocessor(images=image, return_tensors="pt").to(device=self.device, dtype=self.torch_dtype)
        self.model(**inputs)
        return self.activations

    @torch.no_grad()
    def get_activations_from_images(self, images):
        """
        images: a single PIL image or a list of PIL images
        """
        self.activations.clear()
        inputs = self.preprocessor(images=images, return_tensors="pt").to(device=self.device, dtype=self.torch_dtype)
        self.model(**inputs)
        return self.activations
    
    @torch.no_grad()
    def get_activations(self, batch_tokens):
        self.activations.clear()
        self.model(batch_tokens.to(device=self.device, dtype=self.torch_dtype))
        return self.activations

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @torch.no_grad()
    def forward_from_layer(self, image, layer_idx, modified_hidden, token_type='CLS'):
        """
        Forward the CLIP ViT-B-32 image encoder from layer_idx using modified_hidden as input for that layer.
        Returns activations for layer_idx+1 (shape: [1, seq_len, hidden_dim]).
        token_type: 'CLS' or 'IMG'
        """
        # Preprocess image
        inputs = self.preprocessor(images=image, return_tensors="pt").to(device=self.device, dtype=self.torch_dtype)
        pixel_values = inputs['pixel_values']  # [1, 3, 224, 224]

        # Patch embeddings
        patch_emb = self.model.embeddings.patch_embedding(pixel_values)  # [1, hidden_dim, grid, grid]
        patch_emb = patch_emb.flatten(2).transpose(1, 2)  # [1, num_patches, hidden_dim]

        hidden_dim = self.model.embeddings.class_embedding.shape[-1]
        cls_token = self.model.embeddings.class_embedding.unsqueeze(0).unsqueeze(1)  # [1, 1, hidden_dim]
        cls_token = cls_token.expand(patch_emb.shape[0], 1, hidden_dim)  # [1, 1, hidden_dim]

        x = torch.cat((cls_token, patch_emb), dim=1)  # [1, seq_len, hidden_dim]
        position_ids = torch.arange(x.shape[1], dtype=torch.long, device=x.device).unsqueeze(0)  # [1, seq_len]
        pos_embed = self.model.embeddings.position_embedding(position_ids)  # [1, seq_len, hidden_dim]
        x = x + pos_embed
        x = self.model.pre_layrnorm(x)

        # Forward up to layer_idx
        for i, block in enumerate(self.model.encoder.layers):
            if i == layer_idx:
                if token_type == 'CLS':
                    x[:, 0, :] = modified_hidden
                elif token_type == 'IMG':
                    x[:, 1:, :] = modified_hidden
            x = block(x, attention_mask=None, causal_attention_mask=None)
            if i == layer_idx + 1:
                break
        # x now contains the output of layer_idx+1
        return x  # [1, seq_len, hidden_dim]

    def preprocess_image(self, image):
        """
        Preprocess a PIL image using the HuggingFace CLIP processor.
        Returns a dict with 'pixel_values': [1, 3, 224, 224] tensor.
        """
        return self.preprocessor(images=image, return_tensors="pt")
    
    @torch.no_grad()
    def get_activations_from_tensor_batch(self, tensor_batch):
        """
        Process a batch of normalized tensors directly without PIL conversion.
        tensor_batch: [batch_size, 3, 224, 224] normalized tensors
        """
        self.activations.clear()
        
        # If tensors are already normalized for CLIP, use them directly
        # Otherwise, you might need to adjust normalization here
        
        # Process through the model
        if 'clip' in self.model_name:
            # For CLIP, we can pass the tensor directly to the vision model
            outputs = self.model(pixel_values=tensor_batch.to(self.device))
        elif 'dinov2' in self.model_name:
            outputs = self.model(pixel_values=tensor_batch.to(self.device))
        
        return self.activations.copy()
