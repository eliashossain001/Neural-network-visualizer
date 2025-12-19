"""
Procedural Art Generator
Generates unique, beautiful artwork using mathematical algorithms
Each run creates a completely unique piece!
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random
import datetime

class ArtGenerator:
    def __init__(self, seed=None):
        self.seed = seed if seed else random.randint(0, 1000000)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.width = 2000
        self.height = 2000
        
        # Create coordinate system
        x = np.linspace(-2, 2, self.width)
        y = np.linspace(-2, 2, self.height)
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.Theta = np.arctan2(self.Y, self.X)
        
    def perlin_noise(self, scale=1.0, octaves=6):
        """Generate Perlin-like noise"""
        noise = np.zeros_like(self.R)
        amplitude = 1.0
        frequency = scale
        
        for _ in range(octaves):
            noise += amplitude * np.sin(frequency * self.X * np.pi) * np.cos(frequency * self.Y * np.pi)
            amplitude *= 0.5
            frequency *= 2.0
        
        return noise
    
    def flow_field(self, complexity=5):
        """Generate organic flow field patterns"""
        field = np.zeros_like(self.R)
        
        for i in range(complexity):
            freq = (i + 1) * 2
            phase = np.random.random() * 2 * np.pi
            
            pattern = (
                np.sin(freq * self.X + phase) * 
                np.cos(freq * self.Y + phase * 0.7) *
                np.exp(-self.R * 0.5)
            )
            field += pattern / (i + 1)
        
        return field
    
    def mandala_pattern(self, n_petals=12):
        """Generate mandala-like patterns"""
        pattern = np.zeros_like(self.R)
        
        for i in range(n_petals):
            angle = 2 * np.pi * i / n_petals
            rotated_theta = self.Theta - angle
            
            petal = (
                np.sin(n_petals * rotated_theta) * 
                np.exp(-self.R) * 
                np.cos(self.R * 5)
            )
            pattern += petal
        
        # Add radial component
        radial = np.sin(self.R * 10) * np.exp(-self.R * 0.5)
        pattern += radial
        
        return pattern
    
    def quantum_wave(self):
        """Generate quantum interference-like patterns"""
        n_sources = random.randint(3, 8)
        wave = np.zeros_like(self.R)
        
        for _ in range(n_sources):
            cx = random.uniform(-1.5, 1.5)
            cy = random.uniform(-1.5, 1.5)
            freq = random.uniform(5, 15)
            phase = random.uniform(0, 2 * np.pi)
            
            distance = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2)
            wave += np.sin(freq * distance + phase) * np.exp(-distance * 0.5)
        
        return wave
    
    def cellular_automata(self, iterations=50):
        """Generate cellular automata-like patterns"""
        # Initialize random grid
        grid = np.random.choice([0, 1], size=(100, 100), p=[0.6, 0.4])
        
        for _ in range(iterations):
            new_grid = grid.copy()
            for i in range(1, 99):
                for j in range(1, 99):
                    neighbors = grid[i-1:i+2, j-1:j+2].sum() - grid[i, j]
                    if grid[i, j] == 1 and (neighbors < 2 or neighbors > 3):
                        new_grid[i, j] = 0
                    elif grid[i, j] == 0 and neighbors == 3:
                        new_grid[i, j] = 1
            grid = new_grid
        
        # Interpolate to full resolution
        from scipy.ndimage import zoom
        return zoom(grid, self.width // 100)
    
    def create_colormap(self, style='random'):
        """Create beautiful custom colormaps"""
        if style == 'random':
            styles = ['sunset', 'ocean', 'forest', 'cosmic', 'fire']
            style = random.choice(styles)
        
        palettes = {
            'sunset': ['#0d1b2a', '#1b263b', '#415a77', '#778da9', '#e0e1dd'],
            'ocean': ['#03045e', '#023e8a', '#0077b6', '#0096c7', '#00b4d8', '#48cae4'],
            'forest': ['#081c15', '#1b4332', '#2d6a4f', '#40916c', '#52b788', '#74c69d'],
            'cosmic': ['#10002b', '#240046', '#3c096c', '#5a189a', '#7209b7', '#9d4edd'],
            'fire': ['#03071e', '#370617', '#6a040f', '#9d0208', '#d00000', '#dc2f02']
        }
        
        colors = palettes.get(style, palettes['sunset'])
        return LinearSegmentedColormap.from_list('custom', colors)
    
    def generate_art(self, style=None):
        """Generate a piece of art"""
        if style is None:
            styles = ['flow', 'mandala', 'quantum', 'hybrid']
            style = random.choice(styles)
        
        print(f"🎨 Generating {style} art with seed {self.seed}...")
        
        if style == 'flow':
            pattern = self.flow_field(complexity=random.randint(3, 8))
            noise = self.perlin_noise(scale=random.uniform(0.5, 2.0))
            art = pattern + 0.3 * noise
            
        elif style == 'mandala':
            pattern = self.mandala_pattern(n_petals=random.choice([6, 8, 12, 16]))
            noise = self.perlin_noise(scale=0.5, octaves=4)
            art = pattern + 0.2 * noise
            
        elif style == 'quantum':
            art = self.quantum_wave()
            art += 0.2 * self.perlin_noise(scale=1.0)
            
        elif style == 'hybrid':
            flow = self.flow_field(complexity=5)
            quantum = self.quantum_wave()
            mandala = self.mandala_pattern(n_petals=12)
            art = 0.4 * flow + 0.4 * quantum + 0.2 * mandala
        
        # Normalize
        art = (art - art.min()) / (art.max() - art.min())
        
        # Apply non-linear transformation for artistic effect
        art = np.power(art, random.uniform(0.7, 1.5))
        
        return art, style
    
    def save_art(self, art, style):
        """Save the artwork"""
        fig, ax = plt.subplots(figsize=(20, 20), facecolor='black')
        ax.set_position([0, 0, 1, 1])
        ax.axis('off')
        
        cmap = self.create_colormap()
        ax.imshow(art, cmap=cmap, interpolation='bilinear')
        
        # Add signature
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'procedural_art_{style}_{self.seed}_{timestamp}.png'
        
        plt.savefig(filename, dpi=150, facecolor='black', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"✅ Saved: {filename}")
        return filename
    
    def generate_and_save(self, style=None):
        """Generate and save artwork"""
        art, style_used = self.generate_art(style)
        filename = self.save_art(art, style_used)
        
        # Also display
        self.display_art(art, style_used)
        
        return filename
    
    def display_art(self, art, style):
        """Display the artwork"""
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
        ax.axis('off')
        
        cmap = self.create_colormap()
        ax.imshow(art, cmap=cmap, interpolation='bilinear')
        
        title = f'{style.upper()} - Seed: {self.seed}'
        fig.suptitle(title, color='white', fontsize=16, y=0.98)
        
        plt.tight_layout()
        plt.show()

def generate_collection(n=5):
    """Generate a collection of artworks"""
    print(f"🎨 Generating collection of {n} artworks...")
    
    for i in range(n):
        print(f"\nGenerating piece {i+1}/{n}...")
        seed = random.randint(0, 1000000)
        generator = ArtGenerator(seed=seed)
        generator.generate_and_save()
    
    print(f"\n✨ Collection complete! Generated {n} unique pieces.")

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("🎨 PROCEDURAL ART GENERATOR")
    print("=" * 60)
    print("\nGenerating unique digital artwork...")
    print("Each piece is completely unique and unrepeatable!\n")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'collection':
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            generate_collection(n)
        else:
            style = sys.argv[1]
            seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
            generator = ArtGenerator(seed=seed)
            generator.generate_and_save(style=style)
    else:
        # Generate single random piece
        generator = ArtGenerator()
        generator.generate_and_save()
    
    print("\n✨ Done! Your artwork is ready to share!")
