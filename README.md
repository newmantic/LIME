# LIME


LIME is an algorithm designed to explain the predictions of any machine learning model (hence "model-agnostic"). The core idea is that for a particular input, a complex model can be approximated locally (around the neighborhood of the input) by a simpler, interpretable model such as linear regression or decision trees. LIME generates interpretable explanations by learning this simpler model around the instance being explained.


1. Perturbation of Input Data
For a given instance x that we want to explain, LIME generates several perturbed instances x'_i, where each x'_i is a slight variation of x. Perturbations are sampled from the neighborhood around x, and the prediction of the complex model for each perturbed instance is recorded.
Formally, for an input x, generate a set of N perturbations:
X' = {x'_1, x'_2, ..., x'_N}
Where each x'_i is a perturbed version of x.

2. Weighting of Perturbations
To focus more on the instances that are closer to the input x, a weighting function is used that assigns higher weights to perturbations closer to x. A common choice for the weighting function is an exponential kernel based on the Euclidean distance d(x, x'_i) between x and x'_i.
w(x, x'_i) = exp(- (d(x, x'_i))^2 / kernel_width^2)
Where w(x, x'_i) is the weight assigned to perturbation x'_i, and kernel_width is a parameter that controls how quickly the weights decay as the distance increases.

3. Prediction of Perturbations by Complex Model
The complex model f (e.g., a neural network or decision tree) is then used to predict the output for each perturbed instance x'_i. Denote the complex model’s prediction for x'_i as f(x'_i).
Thus, for each perturbed instance, we have:
f(x'_i) for i = 1, 2, ..., N

4. Learning the Local Interpretable Model
LIME then fits a simpler, interpretable model g (often linear regression or decision tree) to the perturbed instances X', using the predictions of the complex model f(x'_i) as the target outputs. The model g is learned in such a way that it approximates the complex model locally, focusing more on the instances that are weighted more heavily (closer to x).
Formally, g is chosen to minimize the following objective function:
L(f, g, w) = sum_{i=1}^{N} w(x, x'_i) * (f(x'_i) - g(x'_i))^2 + Omega(g)
Where:
L(f, g, w) is the loss function.
w(x, x'_i) is the weight of perturbation x'_i.
f(x'_i) is the prediction of the complex model for perturbation x'_i.
g(x'_i) is the prediction of the simpler interpretable model g.
Omega(g) is a regularization term that penalizes the complexity of g.
The goal is to make the simple model g closely match the predictions of the complex model f on the perturbed instances, while keeping the simple model g easy to interpret (which is controlled by the regularization term Omega(g)).

5. Feature Importance (Explanation)
Once the local model g is learned, the coefficients of the linear model (in the case of linear regression) or the feature importance (in the case of decision trees) can be interpreted as the contribution of each feature to the prediction of x. These coefficients or importance values form the explanation for the prediction of the original input x.
Let:
g(x) = beta_0 + beta_1 * x_1 + beta_2 * x_2 + ... + beta_d * x_d
Where beta_1, beta_2, ..., beta_d are the learned coefficients of the simple model g. The values beta_1, beta_2, ..., beta_d provide the feature importance for each feature in the original input x.

