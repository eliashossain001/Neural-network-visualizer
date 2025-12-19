"""
Hypnotic Spiral Generator - Creates mesmerizing animated patterns
Run this and watch the magic! Press 'q' to quit, 's' to save a frame.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import sys

class HypnoticSpiral:
    def __init__(self, resolution=800, num_spirals=12):
        self.resolution = resolution
        self.num_spirals = num_spirals
        self.fig, self.ax = plt.subplots(figsize=(10, 10), facecolor='black')
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.axis('off')
        
        # Create coordinate grid
        x = np.linspace(-2, 2, resolution)
        y = np.linspace(-2, 2, resolution)
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.Theta = np.arctan2(self.Y, self.X)
        
        self.im = None
        self.frame_count = 0
        
    def generate_pattern(self, t):
        """Generate the hypnotic pattern for time t"""
        # Multiple layers of spirals with different frequencies
        pattern = np.zeros_like(self.R)
        
        for i in range(self.num_spirals):
            freq = (i + 1) * 3
            phase = t * (i + 1) * 0.5
            spiral = np.sin(freq * self.Theta + self.R * 10 - phase)
            
            # Add radial waves
            radial = np.sin(self.R * 15 - t * 2 + i * 0.5)
            
            # Combine with rotation
            rotation = np.sin(self.Theta * (i + 1) + t * (i + 1) * 0.3)
            
            pattern += spiral * radial * rotation * (1 / (i + 1))
        
        # Normalize and apply color transformation
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        pattern = np.power(pattern, 1.5)  # Enhance contrast
        
        return pattern
    
    def update(self, frame):
        """Update function for animation"""
        t = frame * 0.05
        pattern = self.generate_pattern(t)
        
        if self.im is None:
            self.im = self.ax.imshow(pattern, cmap='twilight', 
                                    interpolation='bilinear', 
                                    extent=[-2, 2, -2, 2],
                                    vmin=0, vmax=1)
        else:
            self.im.set_array(pattern)
        
        self.frame_count = frame
        return [self.im]
    
    def save_frame(self):
        """Save current frame as image"""
        filename = f'hypnotic_spiral_{self.frame_count:04d}.png'
        self.fig.savefig(filename, facecolor='black', dpi=150, bbox_inches='tight')
        print(f"Saved {filename}")
    
    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'q':
            plt.close()
            sys.exit(0)
        elif event.key == 's':
            self.save_frame()
    
    def animate(self):
        """Start the animation"""
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        anim = FuncAnimation(self.fig, self.update, frames=range(1000),
                           interval=50, blit=True, repeat=True)
        plt.show()

if __name__ == "__main__":
    print("🌀 Starting Hypnotic Spiral Generator...")
    print("Press 's' to save current frame")
    print("Press 'q' to quit")
    
    spiral = HypnoticSpiral(resolution=600, num_spirals=8)
    spiral.animate()
