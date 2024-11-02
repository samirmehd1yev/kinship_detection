import os
import torch
import torch.nn as nn
from torchinfo import summary
import matplotlib.pyplot as plt
import json
from kin_nonkin_training import KinshipConfig, KinshipModel

def analyze_model_architecture(model, save_dir='evaluations/model_analysis'):
    """Analyze and visualize the model architecture"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get model summary using torchinfo
    batch_size = 1
    input_size = (3, 112, 112)
    anchor = torch.randn(batch_size, *input_size)
    other = torch.randn(batch_size, *input_size)
    
    model_summary = summary(
        model,
        input_data=[anchor, other],
        verbose=0,
        depth=10,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
    )
    
    # Save model summary to file
    with open(os.path.join(save_dir, 'model_summary.txt'), 'w') as f:
        print(model_summary, file=f)
    
    # Generate detailed architecture information
    architecture_info = {
        'feature_extractor': {
            'initial_conv': {
                'kernel_size': 7,
                'stride': 2,
                'padding': 3,
                'output_channels': 64
            },
            'residual_layers': [
                {'name': 'layer1', 'in_channels': 64, 'out_channels': 64, 'blocks': 3},
                {'name': 'layer2', 'in_channels': 64, 'out_channels': 128, 'blocks': 4},
                {'name': 'layer3', 'in_channels': 128, 'out_channels': 256, 'blocks': 6},
                {'name': 'layer4', 'in_channels': 256, 'out_channels': 512, 'blocks': 3}
            ],
            'embedding_size': 512
        },
        'fusion_network': {
            'input_size': 1024,
            'hidden_layers': [512, 256],
            'dropout_rate': 0.5
        },
        'kinship_verifier': {
            'input_size': 256,
            'output_size': 1
        }
    }
    
    # Save architecture info as JSON
    with open(os.path.join(save_dir, 'architecture_info.json'), 'w') as f:
        json.dump(architecture_info, f, indent=4)
    
    return architecture_info

def print_model_stats(model):
    """Print model statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Memory usage estimation
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024**2  # MB
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers()) / 1024**2  # MB
    
    print(f"\nApproximate model size:")
    print(f"Parameters: {param_size:.2f} MB")
    print(f"Buffers: {buffer_size:.2f} MB")
    print(f"Total: {(param_size + buffer_size):.2f} MB")

def visualize_model_structure(save_dir='evaluations/model_analysis'):
    """Create a simple visualization of the model structure using matplotlib"""
    plt.figure(figsize=(15, 10))
    
    def add_box(x, y, width, height, label, color='lightblue'):
        plt.gca().add_patch(plt.Rectangle((x, y), width, height, 
                                        facecolor=color, edgecolor='black', alpha=0.3))
        plt.text(x + width/2, y + height/2, label, 
                horizontalalignment='center', verticalalignment='center')
    
    # Feature Extractor
    add_box(0, 0, 3, 8, 'Feature Extractor\n(Shared Weights)', 'lightblue')
    
    # Add components
    y = 7
    for component in ['Conv1 7x7', 'BatchNorm + ReLU', 'MaxPool 3x3',
                     'ResBlock x3\n64→64', 'ResBlock x4\n64→128',
                     'ResBlock x6\n128→256', 'ResBlock x3\n256→512',
                     'Global AvgPool\nFC 512→512']:
        add_box(0.5, y, 2, 0.8, component, 'lightyellow')
        y -= 1
    
    # Fusion Network
    add_box(4, 3, 3, 2, 'Fusion Network\n1024→512→256', 'lightgreen')
    
    # Kinship Verifier
    add_box(4, 1, 3, 1, 'Kinship Verifier\n256→1', 'lightpink')
    
    # Add arrows
    plt.arrow(3, 4, 0.8, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    plt.arrow(5.5, 3, 0, -0.8, head_width=0.1, head_length=0.1, fc='k', ec='k')
    
    plt.xlim(-0.5, 8)
    plt.ylim(0, 8.5)
    plt.axis('off')
    plt.title('Kinship Verification Model Architecture')
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, 'model_structure.png'), bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Create model
    config = KinshipConfig()
    model = KinshipModel(config)
    
    # Create analysis directory
    os.makedirs('evaluations/model_analysis', exist_ok=True)
    
    # Analyze model
    architecture_info = analyze_model_architecture(model)
    print_model_stats(model)
    
    # Create visualization
    visualize_model_structure()
    
    print("\nModel analysis completed! Check the 'evaluations/model_analysis' directory for detailed information.")

if __name__ == "__main__":
    main()