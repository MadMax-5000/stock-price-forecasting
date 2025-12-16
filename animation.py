from manim import *
from feature_engineering import final_df
import numpy as np

df = final_df

class AnimatedGraph(Scene):
    def construct(self):
        # Get your data
        dates = df["Day_of_Week"].values
        prices = df["Close"].values
        
        # Normalize data to fit the axes
        # Map dates to x-axis range
        x_indices = np.arange(len(dates))
        x_min, x_max = 0, len(dates) - 1
        
        # Map prices to y-axis range
        y_min, y_max = prices.min(), prices.max()
        
        # Create axes with appropriate ranges
        axes = Axes(
            x_range=[x_min, x_max, max(1, len(dates) // 10)],
            y_range=[y_min * 0.95, y_max * 1.05, (y_max - y_min) / 10],
            x_length=12,
            y_length=7,
            axis_config={
                "stroke_color": BLUE,
                "stroke_width": 2,
                "include_numbers": True,
            },
            tips=False,
        )
        
        # Create labels
        x_label = axes.get_x_axis_label("Date Index", edge=DOWN, direction=DOWN)
        y_label = axes.get_y_axis_label("Price", edge=LEFT, direction=LEFT)
        
        # Create the line graph
        points = [axes.coords_to_point(i, price) for i, price in enumerate(prices)]
        graph = VMobject()
        graph.set_points_smoothly(points)
        graph.set_stroke(YELLOW, width=3)
        
        # Add everything to the scene
        self.add(axes, x_label, y_label)
        self.play(Create(graph), run_time=3)
        self.wait()
        
        # Optional: Add title
        title = Text(f"Price Over Time ({len(dates)} days)", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(2)