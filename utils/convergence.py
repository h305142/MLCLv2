import torch
import numpy as np
import matplotlib.pyplot as plt

class ConvergenceAnalyzer:
    def __init__(self):
        self.gradient_norms = []
        self.objective_values = []
        self.transport_costs = []
        self.filtering_ratios = []
        
    def update(self, grad_norm, objective, transport_cost, filtering_ratio):
        self.gradient_norms.append(grad_norm)
        self.objective_values.append(objective)
        self.transport_costs.append(transport_cost)
        self.filtering_ratios.append(filtering_ratio)
    
    def check_convergence(self, epsilon=1e-3, window=10):
        if len(self.gradient_norms) < window:
            return False
        
        recent_grad_norms = self.gradient_norms[-window:]
        min_grad_norm = min(recent_grad_norms)
        
        return min_grad_norm <= epsilon**2
    
    def plot_convergence(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0,0].plot(self.gradient_norms)
        axes[0,0].set_title('Gradient Norm')
        axes[0,0].set_yscale('log')
        
        axes[0,1].plot(self.objective_values)
        axes[0,1].set_title('Objective Value')
        
        axes[1,0].plot(self.transport_costs)
        axes[1,0].set_title('Average Transport Cost')
        
        axes[1,1].plot(self.filtering_ratios)
        axes[1,1].set_title('Filtering Ratio')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()