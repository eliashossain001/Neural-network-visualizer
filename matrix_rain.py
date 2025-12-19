"""
Matrix Digital Rain Effect
A cool terminal animation inspired by The Matrix
"""
import random
import time
import sys
import os

class MatrixRain:
    def __init__(self, width=None, height=None):
        # Get terminal size
        if width is None or height is None:
            size = os.get_terminal_size()
            self.width = width or size.columns
            self.height = height or size.lines - 1
        else:
            self.width = width
            self.height = height
        
        # Characters to use (mix of katakana, latin, and numbers)
        self.chars = list("ｦｱｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ012345789Z:.\"=*+-<>¦╌") 
        
        # Initialize streams
        self.streams = []
        for x in range(self.width):
            self.streams.append({
                'x': x,
                'y': random.randint(-self.height, 0),
                'speed': random.uniform(0.3, 1.5),
                'length': random.randint(5, 25),
                'chars': [random.choice(self.chars) for _ in range(30)]
            })
        
        # Color codes
        self.GREEN_BRIGHT = '\033[92m'
        self.GREEN_NORMAL = '\033[32m'
        self.GREEN_DIM = '\033[2;32m'
        self.WHITE = '\033[97m'
        self.RESET = '\033[0m'
        
    def clear_screen(self):
        """Clear the terminal"""
        os.system('clear' if os.name != 'nt' else 'cls')
    
    def hide_cursor(self):
        """Hide terminal cursor"""
        sys.stdout.write('\033[?25l')
        sys.stdout.flush()
    
    def show_cursor(self):
        """Show terminal cursor"""
        sys.stdout.write('\033[?25h')
        sys.stdout.flush()
    
    def move_cursor(self, x, y):
        """Move cursor to position"""
        sys.stdout.write(f'\033[{y};{x}H')
    
    def draw_char(self, x, y, char, color):
        """Draw a character at position with color"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.move_cursor(x + 1, y + 1)
            sys.stdout.write(f'{color}{char}{self.RESET}')
    
    def update_stream(self, stream):
        """Update a single stream"""
        stream['y'] += stream['speed']
        
        # Reset if stream goes off screen
        if stream['y'] - stream['length'] > self.height:
            stream['y'] = random.randint(-20, 0)
            stream['speed'] = random.uniform(0.3, 1.5)
            stream['length'] = random.randint(5, 25)
        
        # Occasionally change characters
        if random.random() < 0.1:
            idx = random.randint(0, len(stream['chars']) - 1)
            stream['chars'][idx] = random.choice(self.chars)
    
    def draw_stream(self, stream):
        """Draw a single stream"""
        for i in range(stream['length']):
            y = int(stream['y'] - i)
            if 0 <= y < self.height:
                char_idx = int(stream['y'] + i) % len(stream['chars'])
                char = stream['chars'][char_idx]
                
                # Color based on position in stream
                if i == 0:
                    color = self.WHITE  # Bright head
                elif i < 3:
                    color = self.GREEN_BRIGHT
                elif i < stream['length'] // 2:
                    color = self.GREEN_NORMAL
                else:
                    color = self.GREEN_DIM
                
                self.draw_char(stream['x'], y, char, color)
    
    def run(self, duration=None):
        """Run the animation"""
        self.clear_screen()
        self.hide_cursor()
        
        try:
            start_time = time.time()
            frame = 0
            
            while True:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Update and draw all streams
                for stream in self.streams:
                    self.update_stream(stream)
                    self.draw_stream(stream)
                
                sys.stdout.flush()
                time.sleep(0.03)  # ~30 FPS
                
                frame += 1
                
                # Occasionally add some random flashes
                if frame % 10 == 0:
                    for _ in range(3):
                        x = random.randint(0, self.width - 1)
                        y = random.randint(0, self.height - 1)
                        char = random.choice(self.chars)
                        self.draw_char(x, y, char, self.WHITE)
        
        except KeyboardInterrupt:
            pass
        finally:
            self.show_cursor()
            self.clear_screen()
            print("Matrix Rain terminated.")

class TypewriterMessage:
    """Display a message with typewriter effect over the matrix rain"""
    
    @staticmethod
    def display(message, delay=0.1):
        """Display message with typewriter effect"""
        os.system('clear' if os.name != 'nt' else 'cls')
        
        # Center the message
        size = os.get_terminal_size()
        lines = message.split('\n')
        start_y = (size.lines - len(lines)) // 2
        
        # Hide cursor
        sys.stdout.write('\033[?25l')
        
        GREEN = '\033[92m'
        RESET = '\033[0m'
        
        for i, line in enumerate(lines):
            x = (size.columns - len(line)) // 2
            sys.stdout.write(f'\033[{start_y + i};{x}H')
            
            for char in line:
                sys.stdout.write(f'{GREEN}{char}{RESET}')
                sys.stdout.flush()
                time.sleep(delay)
            time.sleep(0.3)
        
        time.sleep(2)
        sys.stdout.write('\033[?25h')  # Show cursor

if __name__ == "__main__":
    print("🎬 Matrix Digital Rain")
    print("Press Ctrl+C to exit\n")
    time.sleep(1)
    
    # Show welcome message
    message = """
    W E L C O M E   T O   T H E   M A T R I X
    
    Follow the white rabbit...
    """
    TypewriterMessage.display(message, delay=0.05)
    
    # Run the matrix rain
    rain = MatrixRain()
    rain.run()  # Run indefinitely until Ctrl+C
