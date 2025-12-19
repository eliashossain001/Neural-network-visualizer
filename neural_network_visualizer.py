"""
Real-Time Neural Network Training Visualizer - Enhanced Version
Watch a neural network learn with dynamic visualizations!
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrow
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

class NeuralNetworkVisualizer:
    def __init__(self):
        # Generate spiral dataset
        np.random.seed(42)
        self.n_points = 300
        self.X, self.y = self.make_spiral_data()
        
        # Simple neural network
        self.W1 = np.random.randn(2, 20) * 0.5
        self.b1 = np.zeros(20)
        self.W2 = np.random.randn(20, 2) * 0.5
        self.b2 = np.zeros(2)
        
        self.learning_rate = 0.01
        self.epoch = 0
        self.loss_history = []
        self.accuracy_history = []
        
        # For animation effects
        self.weight_history = []
        self.activation_history = []
        
        # Setup plot
        self.fig = plt.figure(figsize=(18, 6))
        self.fig.subplots_adjust(top=0.88, bottom=0.1, left=0.05, right=0.95, wspace=0.3)
        self.ax1 = plt.subplot(131)
        self.ax2 = plt.subplot(132)
        self.ax3 = plt.subplot(133)
        
        self.setup_plots()
        
        # Create mesh for decision boundary (do once)
        h = 0.5
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        self.xx, self.yy = np.meshgrid(np.arange(x_min, x_max, h),
                                       np.arange(y_min, y_max, h))
        
    def make_spiral_data(self):
        """Create spiral dataset"""
        n = self.n_points // 2
        theta = np.sqrt(np.random.rand(n)) * 2 * np.pi
        
        r_a = 2 * theta + np.pi
        data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
        x_a = data_a + np.random.randn(n, 2) * 0.5
        
        r_b = -2 * theta - np.pi
        data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
        x_b = data_b + np.random.randn(n, 2) * 0.5
        
        X = np.vstack([x_a, x_b])
        y = np.hstack([np.zeros(n), np.ones(n)])
        
        return X, y
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward pass"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X):
        """Backward pass and update"""
        m = X.shape[0]
        y_onehot = np.zeros_like(self.a2)
        y_onehot[np.arange(m), self.y.astype(int)] = 1
        
        dz2 = self.a2 - y_onehot
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0) / m
        
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0) / m
        
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        
        # Calculate loss
        loss = -np.mean(np.log(self.a2[np.arange(m), self.y.astype(int)] + 1e-8))
        self.loss_history.append(loss)
        
        # Store weight magnitudes for visualization
        weight_mag = np.mean(np.abs(self.W1)) + np.mean(np.abs(self.W2))
        self.weight_history.append(weight_mag)
    
    def setup_plots(self):
        """Setup the three subplots"""
        # Decision boundary plot
        self.ax1.set_title('Decision Boundary (Learning...)', fontsize=12, fontweight='bold', pad=10)
        self.ax1.set_xlim(-15, 15)
        self.ax1.set_ylim(-15, 15)
        self.ax1.set_xlabel('Feature 1')
        self.ax1.set_ylabel('Feature 2')
        
        # Network architecture plot
        self.ax2.set_title('Network Activity', fontsize=12, fontweight='bold', pad=10)
        self.ax2.set_xlim(0, 4)
        self.ax2.set_ylim(0, 20)
        self.ax2.axis('off')
        
        # Loss plot
        self.ax3.set_title('Training Metrics', fontsize=12, fontweight='bold', pad=10)
        self.ax3.set_xlabel('Epoch')
        self.ax3.set_ylabel('Value')
        self.ax3.grid(True, alpha=0.3)
        
    def draw_network(self):
        """Draw the network architecture with dynamic weights"""
        self.ax2.clear()
        self.ax2.set_xlim(0, 4)
        self.ax2.set_ylim(0, 20)
        self.ax2.axis('off')
        self.ax2.set_title(f'Network Activity (Epoch {self.epoch})', 
                          fontsize=12, fontweight='bold', pad=10)
        
        # Layer positions
        layers = [2, 20, 2]
        layer_x = [0.5, 2, 3.5]
        
        # Calculate average activations for a sample point
        sample_point = self.X[0:1]  # Take first point
        _ = self.forward(sample_point)
        hidden_activations = self.a1[0]
        output_activations = self.a2[0]
        
        # Normalize activations for visualization
        hidden_norm = hidden_activations / (hidden_activations.max() + 1e-8)
        output_norm = output_activations / (output_activations.max() + 1e-8)
        
        # Draw neurons with activation-based colors
        neurons = []
        
        # Input layer
        for i in range(2):
            y = 9 + i
            neurons.append((layer_x[0], y))
            circle = Circle((layer_x[0], y), 0.2, color='steelblue', 
                          edgecolor='black', linewidth=2, zorder=3)
            self.ax2.add_patch(circle)
            
        # Hidden layer - color by activation
        for i in range(20):
            y = 0.5 + i
            neurons.append((layer_x[1], y))
            # Color intensity based on activation
            intensity = hidden_norm[i]
            color = plt.cm.RdYlGn(intensity)
            size = 0.15 + 0.1 * intensity  # Size varies with activation
            circle = Circle((layer_x[1], y), size, color=color, 
                          edgecolor='black', linewidth=1.5, zorder=3, alpha=0.8)
            self.ax2.add_patch(circle)
            
        # Output layer - color by activation
        for i in range(2):
            y = 9 + i
            neurons.append((layer_x[2], y))
            intensity = output_norm[i]
            color = plt.cm.RdYlGn(intensity)
            size = 0.2 + 0.15 * intensity
            circle = Circle((layer_x[2], y), size, color=color, 
                          edgecolor='black', linewidth=2, zorder=3, alpha=0.9)
            self.ax2.add_patch(circle)
        
        # Draw connections with weight-based intensity
        # Input to hidden (sample connections)
        for i in range(2):
            for j in range(0, 20, 2):  # Sample every other connection
                x1, y1 = neurons[i]
                x2, y2 = neurons[j + 2]
                weight = abs(self.W1[i, j])
                max_weight = abs(self.W1).max()
                alpha = min(0.8, weight / max_weight) if max_weight > 0 else 0.1
                width = 0.5 + 2 * alpha
                color = 'green' if self.W1[i, j] > 0 else 'red'
                self.ax2.plot([x1, x2], [y1, y2], color=color, 
                            alpha=alpha, linewidth=width, zorder=1)
        
        # Hidden to output (sample connections)
        for i in range(0, 20, 2):
            for j in range(2):
                x1, y1 = neurons[i + 2]
                x2, y2 = neurons[j + 22]
                weight = abs(self.W2[i, j])
                max_weight = abs(self.W2).max()
                alpha = min(0.8, weight / max_weight) if max_weight > 0 else 0.1
                width = 0.5 + 2 * alpha
                color = 'green' if self.W2[i, j] > 0 else 'red'
                self.ax2.plot([x1, x2], [y1, y2], color=color, 
                            alpha=alpha, linewidth=width, zorder=1)
        
        # Add labels
        self.ax2.text(layer_x[0], -0.5, 'Input\n(2)', ha='center', fontsize=9, fontweight='bold')
        self.ax2.text(layer_x[1], -0.5, 'Hidden\n(20)', ha='center', fontsize=9, fontweight='bold')
        self.ax2.text(layer_x[2], -0.5, 'Output\n(2)', ha='center', fontsize=9, fontweight='bold')
        
        # Add legend
        self.ax2.text(3.8, 18, 'Activation', fontsize=8, style='italic')
        self.ax2.text(3.8, 17, 'High', fontsize=7, color='green')
        self.ax2.text(3.8, 16, 'Low', fontsize=7, color='red')
    
    def update(self, frame):
        """Update function for animation"""
        # Train for one epoch
        predictions = self.forward(self.X)
        self.backward(self.X)
        self.epoch += 1
        
        accuracy = np.mean((predictions.argmax(axis=1) == self.y).astype(float))
        self.accuracy_history.append(accuracy)
        
        # Update decision boundary with animation effect
        self.ax1.clear()
        self.ax1.set_title(f'Decision Boundary (Epoch {self.epoch}) - Acc: {accuracy:.1%}', 
                          fontsize=12, fontweight='bold', pad=10)
        self.ax1.set_xlabel('Feature 1')
        self.ax1.set_ylabel('Feature 2')
        
        # Calculate predictions for mesh
        Z = self.forward(np.c_[self.xx.ravel(), self.yy.ravel()])
        Z = Z[:, 1].reshape(self.xx.shape)
        
        # Use contourf with animation (changes over time)
        contour = self.ax1.contourf(self.xx, self.yy, Z, levels=20, cmap='RdYlBu', alpha=0.7)
        
        # Add contour lines to show boundary clearly
        self.ax1.contour(self.xx, self.yy, Z, levels=[0.5], colors='black', 
                        linewidths=2, linestyles='--')
        
        # Plot data points with larger markers
        self.ax1.scatter(self.X[self.y == 0, 0], self.X[self.y == 0, 1],
                        c='blue', marker='o', s=60, edgecolors='black', linewidth=1.5,
                        label='Class 0', zorder=3)
        self.ax1.scatter(self.X[self.y == 1, 0], self.X[self.y == 1, 1],
                        c='red', marker='s', s=60, edgecolors='black', linewidth=1.5,
                        label='Class 1', zorder=3)
        
        # Highlight misclassified points with pulsing effect
        predicted_classes = predictions.argmax(axis=1)
        misclassified = predicted_classes != self.y
        if misclassified.any():
            pulse_size = 120 + 30 * np.sin(self.epoch * 0.3)  # Pulsing effect
            self.ax1.scatter(self.X[misclassified, 0], self.X[misclassified, 1],
                           s=pulse_size, facecolors='none', edgecolors='yellow',
                           linewidths=3, zorder=4, label='Misclassified')
        
        self.ax1.legend(loc='upper right')
        
        # Update network architecture
        self.draw_network()
        
        # Update metrics plot with both loss and accuracy
        if len(self.loss_history) > 1:
            self.ax3.clear()
            self.ax3.set_title('Training Metrics', fontsize=12, fontweight='bold', pad=10)
            
            # Create twin axis for accuracy
            ax3_twin = self.ax3.twinx()
            
            # Plot loss
            epochs = range(len(self.loss_history))
            line1 = self.ax3.plot(epochs, self.loss_history, linewidth=2.5, 
                                 color='darkred', label='Loss', marker='o', 
                                 markersize=3, markevery=max(1, len(epochs)//20))
            
            # Plot accuracy
            line2 = ax3_twin.plot(epochs, self.accuracy_history, linewidth=2.5, 
                                 color='darkgreen', label='Accuracy', marker='s',
                                 markersize=3, markevery=max(1, len(epochs)//20))
            
            self.ax3.set_xlabel('Epoch', fontsize=10)
            self.ax3.set_ylabel('Loss', fontsize=10, color='darkred')
            ax3_twin.set_ylabel('Accuracy', fontsize=10, color='darkgreen')
            
            self.ax3.tick_params(axis='y', labelcolor='darkred')
            ax3_twin.tick_params(axis='y', labelcolor='darkgreen')
            
            self.ax3.grid(True, alpha=0.3)
            self.ax3.set_xlim(0, max(len(self.loss_history), 10))
            self.ax3.set_ylim(0, max(self.loss_history) * 1.1)
            ax3_twin.set_ylim(0, 1.05)
            
            # Add combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            self.ax3.legend(lines, labels, loc='center right')
            
            # Add progress indicator
            progress = min(100, self.epoch / 2)  # Assuming ~200 epochs
            self.ax3.text(0.02, 0.98, f'Progress: {progress:.0f}%', 
                         transform=self.ax3.transAxes, fontsize=9,
                         verticalalignment='top', bbox=dict(boxstyle='round', 
                         facecolor='wheat', alpha=0.5))
        
        # Update main title with learning rate info
        self.fig.suptitle(f'Neural Network Training - Epoch {self.epoch} - Accuracy: {accuracy:.2%} - Loss: {self.loss_history[-1]:.3f}',
                         fontsize=14, fontweight='bold', y=0.98)
        
        return []
    
    def animate(self):
        """Start the animation"""
        anim = FuncAnimation(self.fig, self.update, frames=200,
                           interval=50, repeat=False)
        plt.show()

if __name__ == "__main__":
    print(" Enhanced Neural Network Training Visualizer")
    print("=" * 60)
    print("Watch the network learn with dynamic visualizations!")
    print("- Decision boundary morphs as the model learns")
    print("- Neurons light up based on activation strength")
    print("- Connection thickness shows weight magnitude")
    print("- Misclassified points pulse in yellow")
    print("=" * 60)
    
    visualizer = NeuralNetworkVisualizer()
    visualizer.animate()