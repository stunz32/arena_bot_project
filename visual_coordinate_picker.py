#!/usr/bin/env python3
"""
Visual Coordinate Picker for Arena Bot
Interactive tool to draw rectangles on screen and capture precise card coordinates
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageGrab
import json
import os
from datetime import datetime

class VisualCoordinatePicker:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Arena Bot - Visual Coordinate Picker")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2b2b2b')
        
        # State variables
        self.screenshot = None
        self.screenshot_tk = None
        self.canvas = None
        self.rectangles = []
        self.current_rect = None
        self.start_x = None
        self.start_y = None
        self.drawing = False
        
        # Coordinate storage
        self.captured_coordinates = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="🎯 Arena Bot Coordinate Picker", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#2b2b2b')
        title_label.pack(pady=(0, 10))
        
        # Instructions
        instructions = tk.Label(main_frame, 
                               text="1. Click 'Take Screenshot' to capture your screen\n"
                                    "2. Draw rectangles around the 3 arena draft cards\n"
                                    "3. Click 'Save Coordinates' to apply to the bot",
                               font=('Arial', 11), fg='#cccccc', bg='#2b2b2b', justify=tk.LEFT)
        instructions.pack(pady=(0, 10))
        
        # Control buttons frame
        button_frame = tk.Frame(main_frame, bg='#2b2b2b')
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Screenshot button
        self.screenshot_btn = tk.Button(button_frame, text="📸 Take Screenshot", 
                                       command=self.take_screenshot, font=('Arial', 12, 'bold'),
                                       bg='#4CAF50', fg='white', padx=20, pady=5)
        self.screenshot_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear rectangles button
        self.clear_btn = tk.Button(button_frame, text="🗑️ Clear Rectangles", 
                                  command=self.clear_rectangles, font=('Arial', 11),
                                  bg='#FF9800', fg='white', padx=15, pady=5)
        self.clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Save coordinates button
        self.save_btn = tk.Button(button_frame, text="💾 Save Coordinates", 
                                 command=self.save_coordinates, font=('Arial', 12, 'bold'),
                                 bg='#2196F3', fg='white', padx=20, pady=5)
        self.save_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Test coordinates button
        self.test_btn = tk.Button(button_frame, text="🧪 Test Capture", 
                                 command=self.test_capture, font=('Arial', 11),
                                 bg='#9C27B0', fg='white', padx=15, pady=5)
        self.test_btn.pack(side=tk.LEFT)
        
        # Canvas frame with scrollbars
        canvas_frame = tk.Frame(main_frame, bg='#2b2b2b')
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas with scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg='#1e1e1e', highlightthickness=0)
        v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mouse events for rectangle drawing
        self.canvas.bind("<Button-1>", self.start_rectangle)
        self.canvas.bind("<B1-Motion>", self.draw_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.end_rectangle)
        
        # Status label
        self.status_label = tk.Label(main_frame, text="Ready to capture coordinates", 
                                    font=('Arial', 10), fg='#4CAF50', bg='#2b2b2b')
        self.status_label.pack(pady=(10, 0))
        
        # Rectangle info label
        self.rect_info_label = tk.Label(main_frame, text="Rectangles drawn: 0", 
                                       font=('Arial', 10), fg='#cccccc', bg='#2b2b2b')
        self.rect_info_label.pack()
        
    def take_screenshot(self):
        """Capture full screen screenshot"""
        try:
            self.root.withdraw()  # Hide window during screenshot
            
            # Wait a moment for window to hide
            self.root.after(500, self._capture_screen)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to take screenshot: {str(e)}")
            self.root.deiconify()
    
    def _capture_screen(self):
        """Actually capture the screen after window is hidden"""
        try:
            # Capture full screen
            self.screenshot = ImageGrab.grab()
            
            # Calculate display size for canvas (scale down if too large)
            screen_width, screen_height = self.screenshot.size
            max_display_width = 950
            max_display_height = 500
            
            scale_x = max_display_width / screen_width if screen_width > max_display_width else 1
            scale_y = max_display_height / screen_height if screen_height > max_display_height else 1
            self.display_scale = min(scale_x, scale_y)
            
            # Create display version
            display_width = int(screen_width * self.display_scale)
            display_height = int(screen_height * self.display_scale)
            
            display_screenshot = self.screenshot.resize((display_width, display_height), Image.Resampling.LANCZOS)
            self.screenshot_tk = ImageTk.PhotoImage(display_screenshot)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.screenshot_tk)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            # Clear previous rectangles
            self.rectangles = []
            self.captured_coordinates = []
            
            self.status_label.config(text=f"Screenshot captured: {screen_width}x{screen_height}", fg='#4CAF50')
            self.update_rect_info()
            
            self.root.deiconify()  # Show window again
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture screen: {str(e)}")
            self.root.deiconify()
    
    def start_rectangle(self, event):
        """Start drawing a rectangle"""
        if self.screenshot is None:
            messagebox.showwarning("Warning", "Please take a screenshot first!")
            return
            
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.drawing = True
    
    def draw_rectangle(self, event):
        """Draw rectangle as user drags"""
        if not self.drawing:
            return
            
        current_x = self.canvas.canvasx(event.x)
        current_y = self.canvas.canvasy(event.y)
        
        # Delete current rectangle if it exists
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        
        # Draw new rectangle
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, current_x, current_y,
            outline='#FF5722', width=3, fill='', stipple='gray50'
        )
    
    def end_rectangle(self, event):
        """Finish drawing rectangle"""
        if not self.drawing:
            return
            
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        # Ensure rectangle has minimum size
        if abs(end_x - self.start_x) < 10 or abs(end_y - self.start_y) < 10:
            if self.current_rect:
                self.canvas.delete(self.current_rect)
            self.drawing = False
            return
        
        # Calculate actual screen coordinates (scale back up)
        actual_x1 = int(min(self.start_x, end_x) / self.display_scale)
        actual_y1 = int(min(self.start_y, end_y) / self.display_scale)
        actual_x2 = int(max(self.start_x, end_x) / self.display_scale)
        actual_y2 = int(max(self.start_y, end_y) / self.display_scale)
        
        # Store rectangle info
        rect_info = {
            'canvas_rect': self.current_rect,
            'coordinates': (actual_x1, actual_y1, actual_x2 - actual_x1, actual_y2 - actual_y1),
            'display_coords': (self.start_x, self.start_y, end_x, end_y)
        }
        
        self.rectangles.append(rect_info)
        
        # Add rectangle number label
        center_x = (self.start_x + end_x) / 2
        center_y = (self.start_y + end_y) / 2
        text_id = self.canvas.create_text(center_x, center_y, text=str(len(self.rectangles)), 
                                         font=('Arial', 14, 'bold'), fill='#FF5722')
        rect_info['text_id'] = text_id
        
        self.drawing = False
        self.current_rect = None
        
        self.update_rect_info()
        
        # If we have 3 rectangles, suggest saving
        if len(self.rectangles) == 3:
            self.status_label.config(text="Perfect! 3 card regions captured. Ready to save coordinates.", fg='#4CAF50')
    
    def clear_rectangles(self):
        """Clear all drawn rectangles"""
        for rect_info in self.rectangles:
            self.canvas.delete(rect_info['canvas_rect'])
            if 'text_id' in rect_info:
                self.canvas.delete(rect_info['text_id'])
        
        self.rectangles = []
        self.captured_coordinates = []
        self.update_rect_info()
        self.status_label.config(text="Rectangles cleared. Draw new rectangles around cards.", fg='#FF9800')
    
    def update_rect_info(self):
        """Update rectangle count display"""
        count = len(self.rectangles)
        self.rect_info_label.config(text=f"Rectangles drawn: {count}/3")
        
        if count > 0:
            coords_text = "\n".join([f"Card {i+1}: {rect['coordinates']}" for i, rect in enumerate(self.rectangles)])
            self.rect_info_label.config(text=f"Rectangles drawn: {count}/3\n{coords_text}")
    
    def test_capture(self):
        """Test capture regions by extracting and showing them"""
        if not self.rectangles:
            messagebox.showwarning("Warning", "Please draw rectangles first!")
            return
        
        if self.screenshot is None:
            messagebox.showwarning("Warning", "No screenshot available!")
            return
        
        try:
            # Create test captures directory
            test_dir = "test_captures"
            os.makedirs(test_dir, exist_ok=True)
            
            # Extract each rectangle region
            for i, rect_info in enumerate(self.rectangles):
                x, y, w, h = rect_info['coordinates']
                
                # Extract region from original screenshot
                region = self.screenshot.crop((x, y, x + w, y + h))
                
                # Save test capture
                test_path = os.path.join(test_dir, f"test_card_{i+1}.png")
                region.save(test_path)
                
                print(f"Saved test capture {i+1}: {test_path}")
                print(f"Coordinates: x={x}, y={y}, width={w}, height={h}")
            
            self.status_label.config(text=f"Test captures saved to {test_dir}", fg='#4CAF50')
            messagebox.showinfo("Success", f"Test captures saved!\nCheck: {test_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create test captures: {str(e)}")
    
    def save_coordinates(self):
        """Save coordinates to JSON file for use in main bot"""
        if not self.rectangles:
            messagebox.showwarning("Warning", "Please draw rectangles first!")
            return
        
        if len(self.rectangles) != 3:
            result = messagebox.askyesno("Confirm", 
                                       f"You have {len(self.rectangles)} rectangles, but 3 are recommended for arena cards. Continue anyway?")
            if not result:
                return
        
        try:
            # Prepare coordinate data
            coordinates_data = {
                'timestamp': datetime.now().isoformat(),
                'screen_resolution': f"{self.screenshot.width}x{self.screenshot.height}",
                'card_coordinates': [],
                'format_explanation': {
                    'coordinates': 'Each coordinate set is [x, y, width, height]',
                    'x_y': 'Top-left corner of the card region',
                    'width_height': 'Size of the card region to capture'
                }
            }
            
            # Add each rectangle
            for i, rect_info in enumerate(self.rectangles):
                x, y, w, h = rect_info['coordinates']
                coordinates_data['card_coordinates'].append({
                    'card_number': i + 1,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'coordinates_list': [x, y, w, h]
                })
            
            # Save to JSON file
            output_path = "captured_coordinates.json"
            with open(output_path, 'w') as f:
                json.dump(coordinates_data, f, indent=2)
            
            # Also save to settings format for easy integration
            settings_data = {
                'card_coordinates': [rect['coordinates'] for rect in self.rectangles],
                'screen_resolution': f"{self.screenshot.width}x{self.screenshot.height}",
                'captured_timestamp': datetime.now().isoformat()
            }
            
            settings_path = "coordinate_settings.json"
            with open(settings_path, 'w') as f:
                json.dump(settings_data, f, indent=2)
            
            self.status_label.config(text="Coordinates saved successfully!", fg='#4CAF50')
            
            # Show success message with coordinates
            coord_text = "\n".join([f"Card {i+1}: ({rect['coordinates']})" for i, rect in enumerate(self.rectangles)])
            messagebox.showinfo("Success", 
                              f"Coordinates saved to:\n{output_path}\n\n"
                              f"Captured coordinates:\n{coord_text}\n\n"
                              f"These coordinates are now ready to use in the main Arena Bot!")
            
            print("Coordinates saved successfully!")
            print(f"Output file: {output_path}")
            print(f"Settings file: {settings_path}")
            for i, rect_info in enumerate(self.rectangles):
                print(f"Card {i+1}: {rect_info['coordinates']}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save coordinates: {str(e)}")
    
    def run(self):
        """Start the coordinate picker application"""
        print("🎯 Arena Bot Visual Coordinate Picker")
        print("=====================================")
        print("1. Position Hearthstone in Arena Draft mode")
        print("2. Click 'Take Screenshot' to capture your screen")
        print("3. Draw rectangles around the 3 arena draft cards")
        print("4. Click 'Save Coordinates' to save for the main bot")
        print("")
        
        self.root.mainloop()

if __name__ == "__main__":
    picker = VisualCoordinatePicker()
    picker.run()