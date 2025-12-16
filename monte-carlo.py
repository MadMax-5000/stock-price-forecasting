from manim import *
import numpy as np
from scipy.stats import norm

class MonteCarloDeepDive(Scene):
    def construct(self):
        # ---------------------------------------------------------
        # PART 0: DATA GENERATION & SETUP
        # ---------------------------------------------------------
        # We prepare the math first so the animation is pure visual logic.
        
        # Parameters
        T = 100            # Days
        mc_sims = 150      # More sims for a dense "cloud"
        initial_portfolio = 10000
        
        # Synthetic Data (Cholesky Method)
        np.random.seed(420) 
        n_assets = 5
        # Slight positive drift, reasonable volatility
        mean_returns = np.array([0.0006] * n_assets) 
        # Create a covariance matrix
        A = np.random.rand(n_assets, n_assets)
        cov_matrix = np.dot(A, A.transpose()) * 0.0002 
        weights = np.ones(n_assets) / n_assets
        
        # Pre-calculate simulations
        meanM = np.full(shape=(T, len(weights)), fill_value=mean_returns).T
        portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
        L = np.linalg.cholesky(cov_matrix)
        
        for m in range(mc_sims):
            Z = np.random.normal(size=(T, len(weights)))
            dailyReturns = meanM + np.inner(L, Z)
            portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initial_portfolio

        # SORTING for the color gradient (The "Heatmap" effect)
        final_values = portfolio_sims[-1, :]
        sorted_indices = np.argsort(final_values)
        sorted_sims = portfolio_sims[:, sorted_indices]
        
        # Color Palette: Red (Loss) -> Yellow -> Green (Gain) -> Teal (Moon)
        colors = color_gradient([RED_E, ORANGE, YELLOW, GREEN, TEAL], mc_sims)

        # ---------------------------------------------------------
        # PART 1: THE CANVAS (Slower Setup)
        # ---------------------------------------------------------
        
        # A subtle background grid helps the 3b1b aesthetic
        plane = NumberPlane(
            x_range=[0, T, 10],
            y_range=[8000, 13000, 1000],
            background_line_style={"stroke_color": GREY_D, "stroke_width": 1, "stroke_opacity": 0.5}
        ).add_coordinates()
        
        # Clean axes on top
        axes = Axes(
            x_range=[0, T, 10],
            y_range=[8000, 13000, 1000],
            x_length=10,
            y_length=6,
            axis_config={"include_numbers": False, "color": GREY_B},
            tips=False
        )
        
        # Labels
        labels = VGroup(
            axes.get_y_axis_label(Text("Value ($)", font_size=24).rotate(90*DEGREES), edge=LEFT, buff=0.3),
            axes.get_x_axis_label(Text("Time (Days)", font_size=24), edge=DOWN, buff=0.2)
        )
        
        title = Text("Monte Carlo Simulation", font_size=40).to_edge(UP)
        subtitle = Text("Visualizing Uncertainty", font_size=24, color=GREY).next_to(title, DOWN)

        self.play(FadeIn(plane), Create(axes), Write(labels))
        self.play(Write(title), FadeIn(subtitle))
        self.wait(1)

        # ---------------------------------------------------------
        # PART 2: THE SINGLE RANDOM WALK (The "Why")
        # ---------------------------------------------------------
        
        # Narrative text (bottom right)
        narrative = Text("The market is a random walk.", font_size=24, slant=ITALIC).to_corner(DR)
        self.play(Write(narrative))
        
        # Draw ONE line slowly to show the "struggle" of randomness
        # We use the median line for this demo
        mid_idx = mc_sims // 2
        path_data = sorted_sims[:, mid_idx]
        
        line_single = axes.plot_line_graph(
            x_values=np.arange(T),
            y_values=path_data,
            line_color=YELLOW,
            add_vertex_dots=False,
            stroke_width=3
        )
        
        # Animate the path drawing itself nicely
        self.play(Create(line_single), run_time=4, rate_func=linear)
        
        # Add a dot at the end
        end_dot = Dot(axes.c2p(T-1, path_data[-1]), color=YELLOW)
        self.play(FadeIn(end_dot))
        
        # Change narrative
        new_narrative = Text("But this is just ONE possibility.", font_size=24, color=YELLOW).to_corner(DR)
        self.play(Transform(narrative, new_narrative))
        self.wait(2)
        
        # Fade the single line to prepare for the swarm
        self.play(
            line_single.animate.set_stroke(opacity=0.2, width=1),
            FadeOut(end_dot)
        )

        # ---------------------------------------------------------
        # PART 3: THE MULTIVERSE (The Simulation)
        # ---------------------------------------------------------
        
        narrative2 = Text("We simulate thousands of futures...", font_size=24).to_corner(DR)
        self.play(Transform(narrative, narrative2))
        
        # Create VGroup for the swarm
        sim_lines = VGroup()
        for i in range(mc_sims):
            l = axes.plot_line_graph(
                x_values=np.arange(T),
                y_values=sorted_sims[:, i],
                line_color=colors[i],
                add_vertex_dots=False,
                stroke_width=1.5
            )
            l.set_stroke(opacity=0.5) 
            sim_lines.add(l)
            
        # Animate them: First 10 slow, then the rest fast
        self.play(
            LaggedStart(
                *[Create(line) for line in sim_lines[:10]],
                lag_ratio=0.5,
                run_time=3
            )
        )
        
        narrative3 = Text("...to reveal the probability distribution.", font_size=24, color=BLUE).to_corner(DR)
        self.play(Transform(narrative, narrative3))

        # The Explosion
        self.play(
            LaggedStart(
                *[Create(line) for line in sim_lines[10:]],
                lag_ratio=0.005,
                run_time=4
            )
        )
        self.wait(1)

        # ---------------------------------------------------------
        # PART 4: THE BELL CURVE (The Insight)
        # ---------------------------------------------------------
        # This is the "Cool" part. We draw a distribution curve on the side.
        
        # 1. Calculate stats
        final_vals = sorted_sims[-1, :]
        mu, std = norm.fit(final_vals)
        
        # 2. Define the curve function (Gaussian)
        # We need to map this carefully to the screen coordinates
        # It's a vertical curve at x = T
        
        def vertical_gaussian(y):
            # Normal distribution formula
            val = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((y - mu) / std)**2)
            # Scale it up horizontally so it's visible on screen
            return val * 15000 
            
        # Draw the curve on the right side
        distribution_curve = axes.plot_parametric_curve(
            lambda t: [T + vertical_gaussian(t), t, 0],
            t_range=[8000, 13000],
            color=WHITE,
            stroke_opacity=0.8
        )
        
        # Fill the area (Gradient fill for the bell curve)
        # Manim is tricky with parametric fills, so we use many lines
        fill_lines = VGroup()
        for y_val in np.linspace(8000, 13000, 100):
            x_width = vertical_gaussian(y_val)
            if x_width > 0.1: # Only draw if visible
                # Color logic for the curve fill
                c = colors[int(np.interp(y_val, [min(final_vals), max(final_vals)], [0, mc_sims-1]))]
                ln = Line(
                    axes.c2p(T, y_val),
                    axes.c2p(T + x_width, y_val),
                    color=c,
                    stroke_width=2,
                    stroke_opacity=0.4
                )
                fill_lines.add(ln)

        self.play(Create(distribution_curve), run_time=2)
        self.play(FadeIn(fill_lines))
        
        # Add a Dashed Line for the Mean
        mean_line = DashedLine(
            start=axes.c2p(0, mu),
            end=axes.c2p(T+vertical_gaussian(mu), mu),
            color=WHITE
        )
        mean_label = Text(f"Mean: ${int(mu)}", font_size=20).next_to(mean_line, UP, buff=0.1)
        
        self.play(Create(mean_line), Write(mean_label))
        
        narrative4 = Text("The result is a range of probabilities.", font_size=24, color=WHITE).to_corner(DR)
        self.play(Transform(narrative, narrative4))
        self.wait(3)

        # ---------------------------------------------------------
        # PART 5: CLEAN UP & FADE OUT
        # ---------------------------------------------------------
        
        # Group everything and fade out nicely
        everything = VGroup(
            plane, axes, labels, title, subtitle, 
            sim_lines, line_single, 
            distribution_curve, fill_lines, 
            mean_line, mean_label, narrative
        )
        
        self.play(FadeOut(everything), run_time=2)
        
        final_text = Text("Monte Carlo Simulation", font_size=36, weight=THIN)
        self.play(Write(final_text))
        self.wait(1)