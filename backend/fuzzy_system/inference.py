import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl
from typing import Dict, Any, List

class FuzzyPriorityEvaluator:
    def __init__(self, ga_weights: List[float] = None):
        # Define Antecedents (Inputs)
        self.confidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'confidence')
        self.size_ratio = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'size_ratio')
        
        # Define Consequent (Output)
        self.priority = ctrl.Consequent(np.arange(0, 101, 1), 'priority')

        # Membership Functions
        self.confidence['low'] = fuzzy.trimf(self.confidence.universe, [0, 0, 0.5])
        self.confidence['medium'] = fuzzy.trimf(self.confidence.universe, [0.3, 0.6, 0.8])
        self.confidence['high'] = fuzzy.trimf(self.confidence.universe, [0.6, 1.0, 1.0])

        self.size_ratio['small'] = fuzzy.trimf(self.size_ratio.universe, [0, 0, 0.3])
        self.size_ratio['medium'] = fuzzy.trimf(self.size_ratio.universe, [0.2, 0.5, 0.8])
        self.size_ratio['large'] = fuzzy.trimf(self.size_ratio.universe, [0.6, 1.0, 1.0])

        self.priority['low'] = fuzzy.trimf(self.priority.universe, [0, 0, 40])
        self.priority['medium'] = fuzzy.trimf(self.priority.universe, [30, 50, 70])
        self.priority['high'] = fuzzy.trimf(self.priority.universe, [60, 100, 100])

        # Default rule weights unless optimized by GA
        w = ga_weights if ga_weights and len(ga_weights) == 3 else [1.0, 1.0, 1.0]

        # Fuzzy Rules
        rule1 = ctrl.Rule(self.confidence['high'] & self.size_ratio['large'], self.priority['high'])
        rule2 = ctrl.Rule(self.confidence['medium'] | self.size_ratio['medium'], self.priority['medium'])
        rule3 = ctrl.Rule(self.confidence['low'] & self.size_ratio['small'], self.priority['low'])

        # Apply GA weights to rules dynamically
        rule1.weight = w[0]
        rule2.weight = w[1]
        rule3.weight = w[2]

        system = ctrl.ControlSystem([rule1, rule2, rule3])
        self.sim = ctrl.ControlSystemSimulation(system)

    def evaluate(self, confidence: float, size_ratio: float) -> float:
        try:
            self.sim.input['confidence'] = confidence
            self.sim.input['size_ratio'] = size_ratio
            self.sim.compute()
            return round(float(self.sim.output['priority']), 2)
        except Exception:
            # Fallback if defuzzification falls outside ranges
            return round((confidence * 0.6 + size_ratio * 0.4) * 100, 2)
