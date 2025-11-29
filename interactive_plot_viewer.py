# interactive_plot_viewer.py
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

class InteractivePlotViewer:
    def __init__(self, parent, plot_path):
        self.parent = parent
        self.plot_path = plot_path
        
        self.window = tk.Toplevel(parent)
        self.window.title("Interactive Plot Viewer")
        self.window.geometry("1200x800")
        
        # Main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill="x", pady=5)
        
        ttk.Button(controls_frame, text="Zoom In", command=self._zoom_in).pack(side="left", padx=2)
        ttk.Button(controls_frame, text="Zoom Out", command=self._zoom_out).pack(side="left", padx=2)
        ttk.Button(controls_frame, text="Reset View", command=self._reset_view).pack(side="left", padx=2)
        ttk.Button(controls_frame, text="Pan Mode", command=self._toggle_pan).pack(side="left", padx=2)
        
        # Matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Load and display the plot
        self._load_plot()
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, main_frame)
        self.toolbar.update()
        
        # Zoom state
        self.zoom_level = 1.0
        self.pan_mode = False
        
    def _load_plot(self):
        """Load the plot into matplotlib"""
        try:
            # For image plots
            img = Image.open(self.plot_path)
            self.ax.imshow(img)
            self.ax.axis('off')  # Hide axes for image plots
        except Exception as e:
            self.ax.text(0.5, 0.5, f"Error loading plot: {str(e)}", 
                        ha='center', va='center', transform=self.ax.transAxes)
        
    def _zoom_in(self):
        """Zoom in on the plot"""
        self.zoom_level *= 1.2
        self._apply_zoom()
        
    def _zoom_out(self):
        """Zoom out of the plot"""
        self.zoom_level /= 1.2
        self._apply_zoom()
        
    def _reset_view(self):
        """Reset to original view"""
        self.zoom_level = 1.0
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.canvas.draw()
        
    def _apply_zoom(self):
        """Apply zoom transformation"""
        try:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            x_center = np.mean(xlim)
            y_center = np.mean(ylim)
            
            x_range = (xlim[1] - xlim[0]) / self.zoom_level
            y_range = (ylim[1] - ylim[0]) / self.zoom_level
            
            self.ax.set_xlim([x_center - x_range/2, x_center + x_range/2])
            self.ax.set_ylim([y_center - y_range/2, y_center + y_range/2])
            self.canvas.draw()
        except:
            pass
            
    def _toggle_pan(self):
        """Toggle pan mode"""
        self.pan_mode = not self.pan_mode
        if self.pan_mode:
            self.toolbar.pan()
        else:
            self.toolbar.pan()  # Toggles off