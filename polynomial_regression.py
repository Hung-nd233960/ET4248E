# pylint: skip-file, disable=too-many-locals, disable=too-many-arguments, disable=too-many-branches, disable=too-many-statements
# ruff: noqa
# type: ignore

from manim import *
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate noisy sine data
np.random.seed(0)
X = np.sort(2 * np.pi * np.random.rand(100, 1), axis=0)  # Random x in [0, 2π]
y = np.sin(X) + 0.1 * np.random.randn(100, 1)  # Sine with noise


# Create the scene
class PolynomialRegressionScene(Scene):
    def construct(self):
        # Axes for plotting
        axes = Axes(
            x_range=[0, 2 * np.pi],
            y_range=[-2, 2],
            axis_config={"color": BLUE},
        )

        # Create the sine curve
        sine_curve = axes.plot(lambda x: np.sin(x), color=WHITE)
        sine_label = MathTex(r"\sin(x)").next_to(sine_curve, UP)

        # Show the true sine curve
        self.play(Create(axes), Create(sine_curve), Write(sine_label))
        self.wait(1)

        # Polynomial fitting animation
        degrees = [1, 3, 5, 10]
        for d in degrees:
            model, poly, train_err, test_err, y_train_pred, y_test_pred = (
                self.polynomial_regression(d, X, y)
            )
            polynomial_curve = axes.plot(
                lambda x: model.predict(poly.transform([[x]]))[0], color=YELLOW
            )
            poly_label = MathTex(f"Degree {d}").next_to(polynomial_curve, UP)

            self.play(Create(polynomial_curve), Write(poly_label))
            self.wait(1)

        # Train vs Test Error visualization
        error_graph = self.create_error_graph(degrees, X, y)
        self.play(Create(error_graph))
        self.wait(2)

    def polynomial_regression(self, degree, X, y):
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)

        y_pred = model.predict(X_poly)
        train_error = mean_squared_error(y, y_pred)
        return model, poly, train_error, train_error, y_pred, y_pred

    def create_error_graph(self, degrees, X, y):
        errors = []
        for d in degrees:
            _, _, train_err, test_err, _, _ = self.polynomial_regression(d, X, y)
            errors.append((d, train_err, test_err))

        error_axes = Axes(
            x_range=[min(degrees), max(degrees)],
            y_range=[0, max([e[1] for e in errors])],
            axis_config={"color": BLUE},
        )

        train_error_points = [error_axes.c2p(d, e[1]) for e in errors]
        test_error_points = [error_axes.c2p(d, e[2]) for e in errors]

        train_line = Line(
            start=train_error_points[0], end=train_error_points[-1], color=RED
        )
        test_line = Line(
            start=test_error_points[0], end=test_error_points[-1], color=GREEN
        )

        return VGroup(error_axes, train_line, test_line)
