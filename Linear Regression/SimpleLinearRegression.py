import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    def __init__(self, x, y):
        # Initialize training data
        self.x_train = x
        self.y_train = y
        self.m = x.shape[0]  # Number of training examples
        self.x_mean = np.mean(x)  # Mean of x for reference
        self.y_mean = np.mean(y)  # Mean of y for reference
        # Normalize x and y using z-score normalization
        self.x_norm = self.normalize_data_z_score(x)
        self.y_norm = self.normalize_data_z_score(y)
    
    def normalize_data_z_score(self, a):
        """ Normalizes data using z-score normalization """
        std_a = np.std(a)
        mean_a = np.mean(a)
        return (a - mean_a) / std_a  # (data - mean) / std deviation for normalization
            
    def calculate_predictions(self, x, w, b):
        """ Calculates predictions using the linear model y = wx + b """
        return w * x + b
    
    def calculate_cost_MSE(self, y, f_wb):
        """ Calculates Mean Squared Error (MSE) as the cost function """
        sum_diff = np.sum((y - f_wb) ** 2)  # Sum of squared differences
        return sum_diff / self.m  # Average of squared differences
    
    def draw_prediction_plot(self, x, y, f_wb, xLabel, yLabel):
        """ Draws the prediction plot with training data and model prediction line """
        plt.plot(x, f_wb, color="blue", label="Prediction")
        plt.scatter(x, y, color="red", label="Training Data")
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title("Prediction Plot")
        plt.legend()
        plt.show()
        
    def calculate_parameters_gradient_descent(self, x, y, learningRate, iterations, tolerance):
        """ 
        Performs gradient descent to calculate optimal parameters w and b.
        Stops if change in parameters is below the tolerance level.
        """
        
        # Initialization
        w = b = 0  # Start with w and b as 0
        j_wb = np.zeros(iterations)  # Array to store cost at each iteration
        for i in range(iterations):
            # Compute predictions with current w, b
            f_wb = self.calculate_predictions(x, w, b)
            # Calculate cost for current predictions
            j_wb[i] = self.calculate_cost_MSE(y, f_wb)
            # Calculate error between predictions and actual values
            error = y - f_wb
            # Compute gradients (partial derivatives) for w and b
            sum_diff_w = np.sum(x * error)  # Gradient for w
            sum_diff_b = np.sum(error)  # Gradient for b
            
            # Update rules for w and b using the gradient
            deviation_w = (-2 / self.m) * sum_diff_w
            deviation_b = (-2 / self.m) * sum_diff_b
            
            # Update w and b with learning rate
            w_new = w - learningRate * deviation_w
            b_new = b - learningRate * deviation_b
            
            # Convergence check: stop if changes are smaller than tolerance
            if (abs(w - w_new) < tolerance and abs(b - b_new) < tolerance):
                print("Convergence Reached")
                return w_new, b_new, j_wb[:i+1]  # Return cost array up to the convergence point
            
            # Update parameters for the next iteration
            w, b = w_new, b_new
        
        # Return final parameters and cost array
        return w, b, j_wb
    
    def draw_gradient_descent_cost_iterations_plot(self, x, y, xLabel, yLabel):
        """ Plots the cost over iterations to visualize gradient descent progress """
        plt.plot(x, y, color="blue")
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title("Gradient Descent Cost over Iterations")
        plt.show()
    
    def denormalize_parameters(self, w, b):
        """ Denormalizes parameters w and b to match the original scale of data """
        w_denorm = w * (np.std(self.y_train) / np.std(self.x_train))
        b_denorm = (b * np.std(self.y_train)) + np.mean(self.y_train) - w_denorm * np.mean(self.x_train)
        return w_denorm, b_denorm  # Return denormalized w and b


if __name__ == "__main__":
    # Training data: house sizes (x) and corresponding prices (y)
    x_train = np.array([600, 800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000])
    y_train = np.array([150000, 200000, 250000, 300000, 375000, 450000, 500000, 550000, 625000, 750000])
    
    # Create an instance of SimpleLinearRegression with the training data
    linearRegression = SimpleLinearRegression(x_train, y_train)
    
    # Calculate parameters using gradient descent on normalized data
    w, b, j_wb = linearRegression.calculate_parameters_gradient_descent(
        linearRegression.x_norm, linearRegression.y_norm, 0.1, 100, 1e-6
    )
    
    # Plot cost over iterations to check gradient descent progress
    linearRegression.draw_gradient_descent_cost_iterations_plot(
        np.arange(len(j_wb)), j_wb, "Iterations", "Cost"
    )
    
    # Display the calculated parameters
    print(f"Parameters calculated from gradient descent -> w : {w}, b : {b}")
    
    # Calculate predictions using normalized parameters
    f_wb = linearRegression.calculate_predictions(linearRegression.x_norm, w, b)
    # Plot training data and predictions
    linearRegression.draw_prediction_plot(linearRegression.x_norm, linearRegression.y_norm, f_wb, "Size", "Price")
    
    # Predict price for a new house size (x_new = 2700)
    x_new = 2700
    w, b = linearRegression.denormalize_parameters(w, b)  # Denormalize parameters to original scale
    prediction = linearRegression.calculate_predictions(x_new, w, b)
    print(f"Prediction : {prediction}")  # Display the predicted price for size 2700
