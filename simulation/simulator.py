"""
Tellurium-based simulation wrapper for bone microenvironment.
"""

import tellurium as te
import numpy as np
from typing import Dict, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)


class BoneEnvironmentSimulator:
    """
    Simulator for bone microenvironment using Tellurium/SBML.
    """
    
    def __init__(self, antimony_model_path: str):
        """
        Initialize simulator with Antimony model.
        
        Args:
            antimony_model_path: Path to .ant file with bone environment model
        """
        self.antimony_model_path = antimony_model_path
        self.model = None
        self.roadrunner = None
        
        self._load_model()
        logger.info(f"Initialized BoneEnvironmentSimulator with {antimony_model_path}")
    
    def _load_model(self):
        """Load Antimony model into Tellurium."""
        if not os.path.exists(self.antimony_model_path):
            raise FileNotFoundError(
                f"Antimony model file not found: {self.antimony_model_path}"
            )
        
        with open(self.antimony_model_path, 'r') as f:
            antimony_string = f.read()
        
        try:
            self.model = te.loada(antimony_string)
            self.roadrunner = self.model
            logger.debug("Successfully loaded Antimony model")
        except Exception as e:
            logger.error(f"Failed to load Antimony model: {e}")
            raise
    
    def set_parameters(self, param_dict: Dict[str, float]):
        """
        Set model parameters.
        
        Args:
            param_dict: Dictionary of parameter_name → value
        """
        for param_name, value in param_dict.items():
            try:
                self.roadrunner[param_name] = value
                logger.debug(f"Set parameter {param_name} = {value}")
            except Exception as e:
                logger.warning(f"Could not set parameter {param_name}: {e}")
    
    def set_initial_conditions(self, species_dict: Dict[str, float]):
        """
        Set initial species concentrations.
        
        Args:
            species_dict: Dictionary of species_name → concentration
        """
        for species_name, concentration in species_dict.items():
            try:
                self.roadrunner[species_name] = concentration
                logger.debug(f"Set initial [{species_name}] = {concentration}")
            except Exception as e:
                logger.warning(f"Could not set species {species_name}: {e}")
    
    def simulate(self, 
                duration: float = 3600.0,
                num_points: int = 361,
                reset: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run simulation.
        
        Args:
            duration: Simulation duration (seconds)
            num_points: Number of time points
            reset: Reset model to initial conditions before simulation
        
        Returns:
            time_array: Time points (seconds)
            species_dict: Dictionary of species_name → concentration_array
        """
        if reset:
            self.roadrunner.reset()
        
        # Configure integrator for stiff ODE systems
        integrator = self.roadrunner.getIntegrator()
        integrator.setValue('stiff', True)
        integrator.setValue('relative_tolerance', 1e-6)
        integrator.setValue('absolute_tolerance', 1e-9)
        
        try:
            # Run simulation
            result = self.roadrunner.simulate(0, duration, num_points)
            
            # Extract time and species data
            time_array = result[:, 0]
            
            species_dict = {}
            species_names = self.roadrunner.getFloatingSpeciesIds()
            
            for i, species_name in enumerate(species_names):
                # Column 0 is time, species start at column 1
                species_dict[species_name] = result[:, i + 1]
            
            logger.info(f"Simulation completed: {duration}s, {num_points} points")
            
            return time_array, species_dict
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
    
    def get_species_value(self, species_name: str) -> float:
        """Get current value of a species."""
        return self.roadrunner[species_name]
    
    def get_parameter_value(self, param_name: str) -> float:
        """Get current value of a parameter."""
        return self.roadrunner[param_name]
    
    def reset(self):
        """Reset model to initial conditions."""
        self.roadrunner.reset()
        logger.debug("Model reset to initial conditions")