import numpy as np
import matplotlib.pyplot as plt

class MultipleLinearRegression:
    def __init__(self, x, y):
        # Initialize the model with training data
        self.x_train = x  # Training features
        self.y_train = y  # Training target variable
        self.m = x.shape[0]  # Number of training examples
        self.x_mean = np.mean(x, axis=0)  # Mean of training features
        self.y_mean = np.mean(y, axis=0)  # Mean of target variable
        self.x_norm = self.normalize_data_z_score(x)  # Normalized training features
        self.y_norm = self.normalize_data_z_score(y)  # Normalized target variable
    
    def normalize_data_z_score(self, a):
        # Normalize the input data using Z-score normalization
        std_a = np.std(a, axis=0)  # Standard deviation
        mean_a = np.mean(a, axis=0)  # Mean
        return (a - mean_a) / std_a  # Return normalized data
            
    def calculate_predictions(self, x, w, b):
        # Calculate predictions using the linear model
        return np.dot(x, w) + b  # Linear combination of weights and features
    
    def calculate_cost_MSE(self, y, f_wb):
        # Calculate the Mean Squared Error (MSE) cost function
        return np.mean((y - f_wb) ** 2)  # Average of squared differences
    
    def calculate_parameters_gradient_descent(self, x, y, learningRate, iterations, tolerance):
        # Perform gradient descent to optimize weights and bias
        w = np.zeros(x.shape[1])  # Initialize weights
        b = 0  # Initialize bias
        j_wb = np.zeros(iterations)  # Array to store cost for each iteration
        
        for i in range(iterations):
            f_wb = self.calculate_predictions(x, w, b)  # Predictions
            j_wb[i] = self.calculate_cost_MSE(y, f_wb)  # Calculate cost
            error = y - f_wb  # Calculate error
            sum_diff_w = np.dot(x.T, error)  # Gradient for weights
            sum_diff_b = np.sum(error)  # Gradient for bias
            deviation_w = (-2 / self.m) * sum_diff_w  # Update rule for weights
            deviation_b = (-2 / self.m) * sum_diff_b  # Update rule for bias
            
            w_new = w - learningRate * deviation_w  # Update weights
            b_new = b - learningRate * deviation_b  # Update bias
            
            # Check for convergence
            if (np.all(abs(w - w_new) < tolerance) and np.all(abs(b - b_new) < tolerance)):
                print("Convergence Reached")
                return w_new, b_new, j_wb[:i+1]  # Return optimized weights and bias
            
            w, b = w_new, b_new  # Update weights and bias for next iteration
        
        return w, b, j_wb  # Return weights, bias, and cost history
    
    def draw_gradient_descent_cost_iterations_plot(self, x, y, xLabel, yLabel):
        # Plot the cost function over iterations
        plt.plot(x, y, color="blue")  # Plot cost
        plt.xlabel(xLabel)  # X-axis label
        plt.ylabel(yLabel)  # Y-axis label
        plt.title("Gradient Descent Cost over Iterations")  # Plot title
        plt.show()  # Show the plot
    
    def denormalize_parameters(self, w, b):
        # Convert normalized weights and bias back to original scale
        w_denorm = w * (np.std(self.y_train) / np.std(self.x_train, axis=0))  # Scale weights
        b_denorm = (b * np.std(self.y_train)) + np.mean(self.y_train) - np.dot(w_denorm, self.x_mean)  # Scale bias
        return w_denorm, b_denorm  # Return denormalized weights and bias


if __name__ == "__main__":
    # Example training data
    # Weight, Volume
    x_train = np.array([
        [790, 1000],
        [1160, 1200],
        [929, 1000],
        [865, 900],
        [1140, 1500],
        [929, 1000],
        [1109, 1400],
        [1365, 1500],
        [1112, 1500],
        [1150, 1600],
        [980, 1100],
        [990, 1300],
        [1112, 1000],
        [1252, 1600],
        [1326, 1600],
        [1330, 1600],
        [1365, 1600],
        [1280, 2200],
        [1119, 1600],
        [1328, 2000],
        [1584, 1600],
        [1428, 2000],
        [1365, 2100],
        [1415, 1600],
        [1415, 2000],
        [1465, 1500],
        [1490, 2000],
        [1725, 2000],
        [1523, 1600],
        [1705, 2000],
        [1605, 2100],
        [1746, 2000],
        [1235, 1600],
        [1390, 1600],
        [1405, 1600],
        [1395, 2500]
    ])
    
    # Co2 emmission
    y_train = np.array([
        99, 95, 95, 90, 105, 105, 90, 92, 98, 99, 99,
        101, 99, 94, 97, 97, 99, 104, 104, 105, 94, 99,
        99, 99, 99, 102, 104, 114, 109, 114, 115, 117, 104,
        108, 109, 120
    ])
    
    # Create an instance of MultipleLinearRegression
    linearRegression = MultipleLinearRegression(x_train, y_train)
    
    # Perform gradient descent to find optimal parameters
    w, b, j_wb = linearRegression.calculate_parameters_gradient_descent(
        linearRegression.x_norm, linearRegression.y_norm, 0.02, 10000, 1e-6
    )
    
    # Draw the cost plot over iterations
    linearRegression.draw_gradient_descent_cost_iterations_plot(
        np.arange(len(j_wb)), j_wb, "Iterations", "Cost"
    )
    
    # Print the calculated parameters
    print(f"Parameters calculated from gradient descent -> w : {w}, b : {b}")
    
    # Make predictions using the normalized data
    f_wb = linearRegression.calculate_predictions(linearRegression.x_norm, w, b)

    # Print predictions
    print(f"Predictions: {f_wb}")
    
    # Predict for a new house size
    new_house = np.array([[2300, 1300]])  
    w, b = linearRegression.denormalize_parameters(w, b)  # Denormalize parameters
    prediction = linearRegression.calculate_predictions(new_house, w, b)  # Make prediction
    print(f"Prediction: {prediction}")  # Output the prediction
