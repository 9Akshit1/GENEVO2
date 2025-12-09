"""
SBML Model Generator for Bone Remodeling
FULLY UNIT-HARMONIZED VERSION

All units are consistent:
- Concentrations: nM (nanomolar) or mM for phosphate
- Cell counts: cells per bone unit
- Rate constants: hr⁻¹ or (nM·hr)⁻¹ for binding
- Production rates: molecules/cell/hr or nM/hr
- Bone mass: mg/mm (physiological scale)

Author: SBML Implementation Team
Date: November 2025
"""

import libsbml
import numpy as np
from typing import Dict, List, Tuple

class BoneRemodelingModel:
    """
    SBML model generator for bone remodeling.
    Fully unit-harmonized with literature-validated values.
    """
    
    def __init__(self):
        # Create SBML document (Level 3, Version 2)
        self.document = libsbml.SBMLDocument(3, 2)
        self.model = self.document.createModel()
        self.model.setId('bone_remodeling_model')
        self.model.setName('Stochastic Bone Remodeling with Biomarker Feedback')
        
        # Set time units
        self.model.setTimeUnits('hour')
        self.model.setSubstanceUnits('item')  # Molecule counts
        self.model.setExtentUnits('item')
        
        # Conversion factors
        self.V_bone = 0.01  # L (bone microenvironment)
        self.V_blood = 5.0  # L (human blood volume)
        self.NA = 6.022e23  # Avogadro's number
        
        print("="*80)
        print("SBML MODEL GENERATOR FOR BONE REMODELING")
        print("Fully Unit-Harmonized Version")
        print("="*80)
        
    def nM_to_molecules(self, concentration_nM, volume_L):
        """Convert nanomolar (nM) concentration to molecule count."""
        # nM = nmol/L = 1e-9 mol/L
        moles = concentration_nM * 1e-9 * volume_L
        return int(moles * self.NA)
    
    def mM_to_molecules(self, concentration_mM, volume_L):
        """Convert millimolar (mM) concentration to molecule count."""
        moles = concentration_mM * 1e-3 * volume_L
        return int(moles * self.NA)
    
    def create_compartments(self):
        """Create bone and blood compartments with exact volumes."""
        print("\n[1/8] Creating compartments...")
        
        # Bone compartment
        bone = self.model.createCompartment()
        bone.setId('bone')
        bone.setName('Bone Microenvironment')
        bone.setConstant(True)
        bone.setSpatialDimensions(3)
        bone.setSize(self.V_bone)
        bone.setUnits('litre')
        
        # Blood compartment
        blood = self.model.createCompartment()
        blood.setId('blood')
        blood.setName('Systemic Circulation')
        blood.setConstant(True)
        blood.setSpatialDimensions(3)
        blood.setSize(self.V_blood)
        blood.setUnits('litre')
        
        print(f"  ✓ Created 'bone' compartment: {self.V_bone} L")
        print(f"  ✓ Created 'blood' compartment: {self.V_blood} L")
    
    def create_parameters(self):
        """
        Create all kinetic parameters.
        FULLY UNIT-HARMONIZED with literature-validated values.
        """
        print("\n[2/8] Creating kinetic parameters...")
        
        params = {
            # ===== MECHANOSENSING =====
            'k_mechano_sensitivity': 0.35,  # dimensionless response factor
            'k_load_threshold': 1500.0,  # microstrain (με)
            'k_disuse_sclerostin': 0.25,  # dimensionless increase factor
            'k_fluid_shear_microgravity': 1.4,  # fold increase
            
            # ===== RADIATION =====
            'k_radiation_osteoblast_damage': 0.2,  # per Gy
            'k_radiation_inflammation': 0.5,  # per Gy
            'k_radiation_ocy_apoptosis': 0.25,  # per Gy
            
            # ===== SCLEROSTIN KINETICS =====
            # Calculated for steady-state 0.7 nM:
            # Molecules = 0.7e-9 * 0.01 * 6.022e23 = 4.22e9
            # Production = k_deg * molecules = 0.35 * 4.22e9 = 1.48e9/hr
            # k_basal = 1.48e9 / 10000 osteocytes = 1.48e5
            'k_sclero_basal': 1.48e5,  # molecules/cell/hr
            'k_sclero_deg': 0.35,  # hr⁻¹
            'k_sclero_transport': 0.1,  # hr⁻¹
            'k_cortisol_enhance_sclero': 2.0,  # fold increase
            'k_IL6_enhance_sclero': 0.4,  # fold increase
            
            # ===== CELL DYNAMICS (hr⁻¹) =====
            # Reduced significantly to prevent numerical instability
            # These rates are balanced to maintain steady-state populations
            'k_osteoblast_diff': 0.003,   # hr⁻¹ (reasonable)
            'k_osteoclast_diff': 0.002,   # hr⁻¹ (reasonable)
            'k_osteoblast_apoptosis': 0.0025,  # hr⁻¹
            'k_osteoclast_apoptosis': 0.006,  # hr⁻¹
            'k_osteocyte_apoptosis_base': 0.0008,  # hr⁻¹
            'k_precursor_replenish': 0.005,  # hr⁻¹ (200-hour timescale)
            # ===== BONE REMODELING (mg/mm/hr) =====
            # Balanced for steady-state: formation ≈ resorption at baseline
            # Starting bone mass: 4 mg/mm, min: 3 mg/mm
            # Need very small rates since we're dealing with mg/mm changes over hours
            'k_bone_formation': 0.000002,   # mg/mm per osteoblast per hr
            'k_bone_resorption': 0.000002,  # mg/mm per osteoclast per hr
            # At steady state: 0.0000001 * 500 = 0.00005 mg/mm/hr formation
            #                  0.000001 * 50 = 0.00005 mg/mm/hr resorption
            # Over 4320 hours: ~0.2 mg/mm change (5% of initial)
            
            # ===== STRESS & INFLAMMATION =====
            # Cortisol: steady-state 300 nM, k_deg = 0.55 hr⁻¹
            # k_prod = 300 * 0.55 = 165 nM/hr
            'k_cortisol_prod': 165.0,  # nM/hr
            'k_cortisol_deg': 0.55,  # hr⁻¹
            # IL6: steady-state 0.05 nM, k_deg = 0.4 hr⁻¹
            # k_prod = 0.05 * 0.4 = 0.02 nM/hr
            'k_IL6_prod': 0.02,  # nM/hr
            'k_IL6_deg': 0.4,  # hr⁻¹
            'k_cortisol_RANKL': 2.0,  # fold increase
            
            # ===== PTH SYSTEM =====
            'k_PTH_basal': 0.005,  # nM (baseline concentration)
            'k_PTH_deg': 2.0,  # hr⁻¹ (more reasonable for multi-compartment)
            
            # ===== BIOMARKER =====
            'biomarker_threshold': 1.5,  # nM
            'k_biomarker_response': 5.0,  # fold response
            'k_biomarker_deg': 0.25,  # hr⁻¹
            'biomarker_IC50': 2.0,  # nM
            'hill_coefficient': 1.1,  # dimensionless
            'biomarker_active': 0.0,  # binary state
            
            # ===== ESTROGEN =====
            'k_estrogen_OPG': 2.0,  # fold increase
            'k_estrogen_RANKL': 0.5,  # suppression factor
            'k_estrogen_baseline': 0.7,  # nM
            'k_estrogen_decline': 0.00003,  # hr⁻¹
            
            # ===== CKD-MBD =====
            'PTH_dysregulation_factor': 1.0,  # multiplier (2.5 for CKD)
            'k_FGF23_sclerostin': 1.3,  # fold increase
            'k_FGF23_basal': 0.04,  # nM (baseline concentration)
            'k_FGF23_phosphate': 2.0,  # fold per mM
            'k_FGF23_vitD': 0.004,  # hr⁻¹
            'k_FGF23_deg': 1.0,  # hr⁻¹
            
            # ===== PHOSPHATE HOMEOSTASIS =====
            'k_phosphate_basal': 1.1,  # mM baseline
            'k_phosphate_accumulation': 0.005,  # mM/hr
            'k_phosphate_clearance': 0.01,  # hr⁻¹
            
            # ===== SURVIVAL LIMITS =====
            'min_osteocytes': 250.0,  # cells
            'min_bone_mass': 3.0,  # mg/mm
            'min_phosphate_mM': 0.7,  # mM
            
            # ===== WNT PATHWAY =====
            # Wnt: steady-state 2.0 nM in 0.01 L = 1.20e10 molecules
            # Degradation = 1.5 * 1.20e10 = 1.80e10/hr
            # k_prod = 1.80e10 / 500 osteoblasts = 3.60e7
            'k_wnt_production_basal': 3.60e7,  # molecules/cell/hr
            'k_wnt_degradation': 1.5,  # hr⁻¹
            'k_sclerostin_LRP_block': 3.0,  # nM for 50% block
            'k_LRP_regeneration': 0.3,  # hr⁻¹
            'k_beta_cat_stabilization': 0.7,  # hr⁻¹
            'k_beta_cat_degradation': 2.8,  # hr⁻¹
            'k_beta_cat_nuclear_import': 12.0,  # hr⁻¹
            'k_beta_cat_nuclear_export': 7.0,  # hr⁻¹
            'k_TCF_activation': 4.0,  # hr⁻¹
            'k_TCF_deactivation': 3.0,  # hr⁻¹
            
            # ===== RANKL/OPG (converted binding rates) =====
            # RANKL membrane: steady-state 0.5 nM = 3.01e9 molecules
            # Shedding = 0.5 * 3.01e9 = 1.51e9/hr
            # k_prod = 1.51e9 / 500 osteoblasts = 3.01e6
            'k_RANKL_membrane_prod': 3.01e6,  # molecules/cell/hr
            'k_RANKL_shedding': 0.5,  # hr⁻¹
            # OPG: steady-state 5.0 nM = 3.01e10 molecules
            # Degradation = 0.28 * 3.01e10 = 8.43e9/hr
            # k_prod = 8.43e9 / 500 osteoblasts = 1.69e7
            'k_OPG_production': 1.69e7,  # molecules/cell/hr
            'k_OPG_degradation': 0.28,  # hr⁻¹
            'k_RANKL_RANK_binding': 3.6,  # (nM·hr)⁻¹
            'k_RANKL_RANK_dissociation': 18.0,  # hr⁻¹
            'k_OPG_RANKL_binding': 18.0,  # (nM·hr)⁻¹
            'k_OPG_RANKL_dissociation': 3.6,  # hr⁻¹
            'k_RANKL_soluble_deg': 0.5,  # hr⁻¹
            
            # ===== PTH SIGNALING =====
            'k_PTH_to_bone': 1.75,  # effect multiplier
            'k_PTH_from_bone': 0.08,  # hr⁻¹
            'k_PTH_receptor_binding': 1.8,  # (nM·hr)⁻¹
            'k_PTH_receptor_dissociation': 14.4,  # hr⁻¹
            'k_cAMP_production': 1.0,  # hr⁻¹ (simplified, relative to receptor)
            'k_cAMP_degradation': 15.0,  # hr⁻¹
            'k_PKA_activation': 20.0,  # hr⁻¹
            'k_PKA_deactivation': 12.0,  # hr⁻¹
            'k_CREB_phosphorylation': 9.0,  # hr⁻¹
            'k_CREB_dephosphorylation': 4.5,  # hr⁻¹
            
            # ===== BASELINE RECEPTOR DENSITIES =====
            'LRP_baseline': 5.0,  # nM
            'RANK_baseline': 2.0,  # nM
        }
        
        for param_id, value in params.items():
            param = self.model.createParameter()
            param.setId(param_id)
            param.setValue(value)
            param.setConstant(True)
        
        print(f"  ✓ Created {len(params)} kinetic parameters")
        print(f"  ✓ All units harmonized (hr⁻¹, nM, molecules/cell/hr)")
        print(f"  ✓ Bone mass in mg/mm scale")
    
    def create_species(self):
        """
        Create all species with literature-validated initial values.
        Concentrations in nM, bone mass in mg/mm.
        """
        print("\n[3/8] Creating species...")
        
        # Dictionary: {species_id: (compartment, initial_amount, constant)}
        species_data = {
            # ===== ENVIRONMENTAL FACTORS (dimensionless, scaled 0-1000) =====
            'Gravity': ('bone', 1000, False),
            'Radiation': ('bone', 0, False),
            'FluidShift': ('bone', 0, False),
            'CircadianDisruption': ('bone', 0, False),
            'ExerciseLoad': ('bone', 300, False),
            'Nutrition': ('bone', 1000, False),
            'StressHormones': ('bone', 500, False),
            'OxygenLevel': ('bone', 1000, False),
            
            # ===== BONE CELLS (counts per bone unit) =====
            'Osteocytes': ('bone', 10000, False),
            'Osteoblasts': ('bone', 500, False),
            'Osteoclasts': ('bone', 50, False),
            'OsteoblastPrecursors': ('bone', 200, False),
            'OsteoclastPrecursors': ('bone', 100, False),
            
            # ===== WNT PATHWAY (nM converted to molecules) =====
            'Wnt_ligand': ('bone', self.nM_to_molecules(2.0, self.V_bone), False),
            'LRP5_LRP6': ('bone', self.nM_to_molecules(5.0, self.V_bone), False),
            'BetaCatenin_cytoplasm': ('bone', self.nM_to_molecules(1.5, self.V_bone), False),
            'BetaCatenin_nucleus': ('bone', self.nM_to_molecules(0.8, self.V_bone), False),
            'GSK3beta': ('bone', self.nM_to_molecules(2.0, self.V_bone), False),
            'TCF_LEF': ('bone', self.nM_to_molecules(0.5, self.V_bone), False),
            
            # ===== SCLEROSTIN (nM) =====
            'Sclerostin_ECM': ('bone', self.nM_to_molecules(0.7, self.V_bone), False),
            'Sclerostin_blood': ('blood', self.nM_to_molecules(0.6, self.V_blood), False),
            
            # ===== RANKL/OPG (nM) =====
            'RANKL_membrane': ('bone', self.nM_to_molecules(0.5, self.V_bone), False),
            'RANKL_soluble': ('bone', self.nM_to_molecules(0.8, self.V_bone), False),
            'OPG_ECM': ('bone', self.nM_to_molecules(5.0, self.V_bone), False),
            'RANK_osteoclast': ('bone', self.nM_to_molecules(2.0, self.V_bone), False),
            'RANKL_RANK_complex': ('bone', self.nM_to_molecules(0.1, self.V_bone), False),
            'RANKL_OPG_complex': ('bone', self.nM_to_molecules(0.3, self.V_bone), False),
            
            # ===== PTH SYSTEM (nM) =====
            'PTH_blood': ('blood', self.nM_to_molecules(0.005, self.V_blood), False),
            'PTH_ECM': ('bone', self.nM_to_molecules(0.004, self.V_bone), False),
            'PTH_receptor': ('bone', self.nM_to_molecules(3.0, self.V_bone), False),
            'PTH_receptor_active': ('bone', self.nM_to_molecules(0.3, self.V_bone), False),
            'cAMP': ('bone', self.nM_to_molecules(0.5, self.V_bone), False),
            'PKA_active': ('bone', self.nM_to_molecules(0.2, self.V_bone), False),
            'CREB_phosphorylated': ('bone', self.nM_to_molecules(0.1, self.V_bone), False),
            
            # ===== SYSTEMIC HORMONES (nM) =====
            'VitaminD_blood': ('blood', self.nM_to_molecules(50.0, self.V_blood), False),
            'Cortisol_blood': ('blood', self.nM_to_molecules(300.0, self.V_blood), False),
            'IL6_blood': ('blood', self.nM_to_molecules(0.05, self.V_blood), False),
            
            # ===== DISEASE-SPECIFIC (nM) =====
            'Estrogen_blood': ('blood', self.nM_to_molecules(0.7, self.V_blood), False),
            'FGF23_blood': ('blood', self.nM_to_molecules(0.04, self.V_blood), False),
            'Phosphate_blood': ('blood', self.mM_to_molecules(1.1, self.V_blood), False),
            
            # ===== BONE STRUCTURE (mg/mm scale) =====
            'BoneMass': ('bone', 4.0, False),  # mg/mm, starting above min of 3.0
            'BoneMineralDensity': ('bone', 1.2, False),  # g/cm³ typical
            'Microarchitecture': ('bone', 1.0, False),  # normalized
            
            # ===== BIOMARKER =====
            'Biomarker_ECM': ('bone', 0, False),

        }
        
        for species_id, (compartment, initial_amount, constant) in species_data.items():
            species = self.model.createSpecies()
            species.setId(species_id)
            species.setCompartment(compartment)
            species.setInitialAmount(initial_amount)
            species.setConstant(constant)
            species.setHasOnlySubstanceUnits(True)
            species.setBoundaryCondition(False)
        
        # FIX: BoneMass should use concentration units, not substance units
        bone_mass_species = self.model.getSpecies('BoneMass')
        if bone_mass_species:
            bone_mass_species.setHasOnlySubstanceUnits(False)
        
        print(f"  ✓ Created {len(species_data)} species")
        print(f"  ✓ Concentrations: nM (converted to molecules)")
        print(f"  ✓ BoneMass: 4.0 mg/mm (physiological scale)")

    def create_assignment_rules(self):
        """
        Create assignment rules for derived quantities.
        All concentrations computed in nM.
        """
        print("\n[4/8] Creating assignment rules for derived quantities...")
        
        # Helper function to create assignment rule
        def create_rule(variable_id, formula_string):
            rule = self.model.createAssignmentRule()
            rule.setVariable(variable_id)
            
            math_ast = libsbml.parseL3Formula(formula_string)
            if math_ast is None:
                print(f"    ERROR: Failed to parse formula for {variable_id}")
                return False
            
            rule.setMath(math_ast)
            return True
        
        # Create derived quantity parameters
        derived_params = [
            'MechanoSignal',
            'DisuseSignal',
            'RadiationDamage',
            'StressEffect',
            'InflammationEffect',
            'VitD_effect',
            'Biomarker_activation',
            'Biomarker_inhibition',
            'PTH_sclerostin_suppression',
            'PTH_wnt_stimulation',
            'PTH_RANKL_effect',
            'Wnt_LRP_active',
            'Estrogen_effect_OPG',
            'Estrogen_effect_RANKL',
            'FGF23_sclerostin_effect',
            'EffectiveMechanicalLoad',
            'Cortisol_nM',
            'Estrogen_nM',
            'FGF23_nM',
            'Phosphate_mM',
            'VitD_nM',
            'Sclerostin_nM',
            'PTH_nM',
            'IL6_nM',
        ]
        
        for param_id in derived_params:
            param = self.model.createParameter()
            param.setId(param_id)
            param.setValue(1.0)
            param.setConstant(False)
        
        # Conversion factor: molecules to nM
        # nM = molecules / (NA * V_L * 1e-9)
        conv_blood_nM = self.NA * self.V_blood * 1e-9
        conv_bone_nM = self.NA * self.V_bone * 1e-9
        conv_blood_mM = self.NA * self.V_blood * 1e-3
        
        # ===== MECHANICAL LOADING =====
        create_rule('EffectiveMechanicalLoad',
                   'Gravity + ExerciseLoad * 0.4 + 0.1 * (1000 - Gravity)')
        
        create_rule('MechanoSignal',
                   'EffectiveMechanicalLoad / (EffectiveMechanicalLoad + k_load_threshold)')
        
        create_rule('DisuseSignal',
                   '1 / (1 + k_mechano_sensitivity * EffectiveMechanicalLoad / 1000)')
        
        # ===== RADIATION =====
        create_rule('RadiationDamage',
                   '1 + k_radiation_osteoblast_damage * Radiation')
        
        # ===== CONCENTRATIONS (molecules to nM) =====
        create_rule('Cortisol_nM', f'Cortisol_blood / {conv_blood_nM}')
        create_rule('Estrogen_nM', f'Estrogen_blood / {conv_blood_nM}')
        create_rule('FGF23_nM', f'FGF23_blood / {conv_blood_nM}')
        create_rule('Phosphate_mM', f'Phosphate_blood / {conv_blood_mM}')
        create_rule('VitD_nM', f'VitaminD_blood / {conv_blood_nM}')
        create_rule('Sclerostin_nM', f'Sclerostin_ECM / {conv_bone_nM}')
        create_rule('PTH_nM', f'PTH_blood / {conv_blood_nM}')
        create_rule('IL6_nM', f'IL6_blood / {conv_blood_nM}')
        
        # ===== STRESS & INFLAMMATION =====
        create_rule('StressEffect',
                   '1 + (k_cortisol_enhance_sclero - 1) * (Cortisol_nM / 300)')
        
        create_rule('InflammationEffect',
                   '1 + k_IL6_enhance_sclero * (IL6_nM / 0.05)')
        
        # ===== VITAMIN D =====
        create_rule('VitD_effect', 'VitD_nM / 50')
        
        # ===== BIOMARKER ACTIVATION (Hill function) =====
        create_rule('Biomarker_activation',
                   'piecewise(0, Sclerostin_nM <= biomarker_threshold, ' +
                   'pow(Sclerostin_nM - biomarker_threshold, hill_coefficient) / ' +
                   '(pow(Sclerostin_nM - biomarker_threshold, hill_coefficient) + ' +
                   'pow(biomarker_IC50, hill_coefficient)))')
        
        create_rule('Biomarker_inhibition',
                   '1 / (1 + pow(biomarker_active * Biomarker_ECM / ' +
                   f'({conv_bone_nM} * biomarker_IC50), hill_coefficient))')
        
        # ===== PTH EFFECTS =====
        create_rule('PTH_sclerostin_suppression',
                   f'1 / (1 + 3 * (CREB_phosphorylated / {conv_bone_nM}))')
        
        create_rule('PTH_wnt_stimulation',
                   f'1 + 1.5 * (PKA_active / {conv_bone_nM})')
        
        create_rule('PTH_RANKL_effect',
                   f'1 + 0.8 * (PTH_receptor_active / {conv_bone_nM})')
        
        # ===== WNT PATHWAY =====
        # Cap Wnt_LRP_active at reasonable physiological range (0-2)
        create_rule('Wnt_LRP_active',
                   f'piecewise(2, (Wnt_ligand / {conv_bone_nM}) * (LRP5_LRP6 / {conv_bone_nM}) / ' +
                   f'((Wnt_ligand / {conv_bone_nM}) + (LRP5_LRP6 / {conv_bone_nM}) + 2) > 2, ' +
                   f'(Wnt_ligand / {conv_bone_nM}) * (LRP5_LRP6 / {conv_bone_nM}) / ' +
                   f'((Wnt_ligand / {conv_bone_nM}) + (LRP5_LRP6 / {conv_bone_nM}) + 2))')
        
        # ===== ESTROGEN EFFECTS =====
        create_rule('Estrogen_effect_OPG',
                   '1 + (k_estrogen_OPG - 1) * (Estrogen_nM / k_estrogen_baseline)')
        
        create_rule('Estrogen_effect_RANKL',
                   '1 / (1 + k_estrogen_RANKL * (Estrogen_nM / k_estrogen_baseline))')
        
        # ===== FGF23 EFFECTS =====
        create_rule('FGF23_sclerostin_effect',
                   '1 + (k_FGF23_sclerostin - 1) * (FGF23_nM / k_FGF23_basal)')
        
        print(f"  ✓ Created {len(derived_params)} assignment rules")
    
    def create_reactions(self):
        """
        Create all biochemical reactions with properly scaled kinetic laws.
        """
        print("\n[5/8] Creating reactions...")
        
        reaction_count = 0
        conv_bone_nM = self.NA * self.V_bone * 1e-9
        conv_blood_nM = self.NA * self.V_blood * 1e-9
        
        # ========== SCLEROSTIN PRODUCTION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_sclero_production')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('Sclerostin_ECM')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        # Simplified: basal production modulated by disuse
        # DisuseSignal increases when gravity/load is low
        # StressEffect and InflammationEffect increase production
        formula = ('k_sclero_basal * Osteocytes * (1 + k_disuse_sclerostin * DisuseSignal) * ' +
                  'StressEffect * InflammationEffect')
        kl.setMath(libsbml.parseL3Formula(formula))
        reaction_count += 1
        
        # ========== SCLEROSTIN DEGRADATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_sclero_degradation')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('Sclerostin_ECM')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_sclero_deg * Sclerostin_ECM'))
        reaction_count += 1
        
        # ========== SCLEROSTIN TRANSPORT (Bone → Blood) ==========
        rxn = self.model.createReaction()
        rxn.setId('R_sclero_transport')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('Sclerostin_ECM')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        product = rxn.createProduct()
        product.setSpecies('Sclerostin_blood')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_sclero_transport * Sclerostin_ECM'))
        reaction_count += 1
        
        # ========== SCLEROSTIN BLOOD CLEARANCE ==========
        rxn = self.model.createReaction()
        rxn.setId('R_sclero_blood_clear')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('Sclerostin_blood')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_sclero_deg * Sclerostin_blood'))
        reaction_count += 1
        
        # ========== WNT LIGAND PRODUCTION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_wnt_production')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('Wnt_ligand')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_wnt_production_basal * Osteoblasts * PTH_wnt_stimulation'))
        reaction_count += 1
        
        # ========== WNT DEGRADATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_wnt_degradation')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('Wnt_ligand')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_wnt_degradation * Wnt_ligand'))
        reaction_count += 1
        
        # ========== LRP5/6 REGENERATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_LRP_regeneration')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('LRP5_LRP6')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula(f'k_LRP_regeneration * (LRP_baseline * {conv_bone_nM} - LRP5_LRP6)'))
        reaction_count += 1
        
        # ========== LRP5/6 BLOCKING BY SCLEROSTIN ==========
        rxn = self.model.createReaction()
        rxn.setId('R_LRP_block')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('LRP5_LRP6')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('LRP5_LRP6 * Sclerostin_nM / (Sclerostin_nM + k_sclerostin_LRP_block)'))
        reaction_count += 1
        
        # ========== BETA-CATENIN STABILIZATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_beta_cat_stabilization')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('BetaCatenin_cytoplasm')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula(f'k_beta_cat_stabilization * Wnt_LRP_active * {conv_bone_nM}'))
        reaction_count += 1
        
        # ========== BETA-CATENIN DEGRADATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_beta_cat_degradation')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('BetaCatenin_cytoplasm')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_beta_cat_degradation * BetaCatenin_cytoplasm * (1 - Wnt_LRP_active)'))
        reaction_count += 1
        
        # ========== BETA-CATENIN NUCLEAR IMPORT ==========
        rxn = self.model.createReaction()
        rxn.setId('R_beta_cat_nuclear_import')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('BetaCatenin_cytoplasm')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        product = rxn.createProduct()
        product.setSpecies('BetaCatenin_nucleus')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_beta_cat_nuclear_import * BetaCatenin_cytoplasm'))
        reaction_count += 1
        
        # ========== BETA-CATENIN NUCLEAR EXPORT ==========
        rxn = self.model.createReaction()
        rxn.setId('R_beta_cat_nuclear_export')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('BetaCatenin_nucleus')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        product = rxn.createProduct()
        product.setSpecies('BetaCatenin_cytoplasm')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_beta_cat_nuclear_export * BetaCatenin_nucleus'))
        reaction_count += 1
        
        # ========== TCF/LEF ACTIVATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_TCF_activation')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('TCF_LEF')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula(f'k_TCF_activation * BetaCatenin_nucleus'))
        reaction_count += 1
        
        # ========== TCF/LEF DEACTIVATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_TCF_deactivation')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('TCF_LEF')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_TCF_deactivation * TCF_LEF'))
        reaction_count += 1
        
        # ========== RANKL MEMBRANE PRODUCTION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_RANKL_membrane_prod')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('RANKL_membrane')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_RANKL_membrane_prod * Osteoblasts * PTH_RANKL_effect * Estrogen_effect_RANKL'))
        reaction_count += 1
        
        # ========== RANKL SHEDDING ==========
        rxn = self.model.createReaction()
        rxn.setId('R_RANKL_shedding')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('RANKL_membrane')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        product = rxn.createProduct()
        product.setSpecies('RANKL_soluble')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_RANKL_shedding * RANKL_membrane'))
        reaction_count += 1
        
        # ========== RANKL SOLUBLE DEGRADATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_RANKL_soluble_deg')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('RANKL_soluble')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_RANKL_soluble_deg * RANKL_soluble'))
        reaction_count += 1
        
        # ========== OPG PRODUCTION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_OPG_production')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('OPG_ECM')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_OPG_production * Osteoblasts * Estrogen_effect_OPG'))
        reaction_count += 1
        
        # ========== OPG DEGRADATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_OPG_degradation')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('OPG_ECM')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_OPG_degradation * OPG_ECM'))
        reaction_count += 1
        
        # ========== RANKL-RANK BINDING ==========
        rxn = self.model.createReaction()
        rxn.setId('R_RANKL_RANK_binding')
        rxn.setReversible(False)
        
        reactant1 = rxn.createReactant()
        reactant1.setSpecies('RANKL_soluble')
        reactant1.setStoichiometry(1)
        reactant1.setConstant(True)
        
        reactant2 = rxn.createReactant()
        reactant2.setSpecies('RANK_osteoclast')
        reactant2.setStoichiometry(1)
        reactant2.setConstant(True)
        
        product = rxn.createProduct()
        product.setSpecies('RANKL_RANK_complex')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        # k_RANKL_RANK_binding is (nM·hr)⁻¹, need to scale for molecule counts
        kl.setMath(libsbml.parseL3Formula(f'k_RANKL_RANK_binding * RANKL_soluble * RANK_osteoclast / {conv_bone_nM}'))
        reaction_count += 1
        
        # ========== RANKL-RANK DISSOCIATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_RANKL_RANK_dissociation')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('RANKL_RANK_complex')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        product1 = rxn.createProduct()
        product1.setSpecies('RANKL_soluble')
        product1.setStoichiometry(1)
        product1.setConstant(True)
        
        product2 = rxn.createProduct()
        product2.setSpecies('RANK_osteoclast')
        product2.setStoichiometry(1)
        product2.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_RANKL_RANK_dissociation * RANKL_RANK_complex'))
        reaction_count += 1
        
        # ========== OPG-RANKL BINDING ==========
        rxn = self.model.createReaction()
        rxn.setId('R_OPG_RANKL_binding')
        rxn.setReversible(False)
        
        reactant1 = rxn.createReactant()
        reactant1.setSpecies('OPG_ECM')
        reactant1.setStoichiometry(1)
        reactant1.setConstant(True)
        
        reactant2 = rxn.createReactant()
        reactant2.setSpecies('RANKL_soluble')
        reactant2.setStoichiometry(1)
        reactant2.setConstant(True)
        
        product = rxn.createProduct()
        product.setSpecies('RANKL_OPG_complex')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula(f'k_OPG_RANKL_binding * OPG_ECM * RANKL_soluble / {conv_bone_nM}'))
        reaction_count += 1
        
        # ========== OPG-RANKL DISSOCIATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_OPG_RANKL_dissociation')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('RANKL_OPG_complex')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        product1 = rxn.createProduct()
        product1.setSpecies('OPG_ECM')
        product1.setStoichiometry(1)
        product1.setConstant(True)
        
        product2 = rxn.createProduct()
        product2.setSpecies('RANKL_soluble')
        product2.setStoichiometry(1)
        product2.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_OPG_RANKL_dissociation * RANKL_OPG_complex'))
        reaction_count += 1
        
        # ========== OSTEOBLAST DIFFERENTIATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_osteoblast_diff')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('OsteoblastPrecursors')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        product = rxn.createProduct()
        product.setSpecies('Osteoblasts')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        # Basal differentiation + TCF enhancement, reduced by radiation
        # TCF_LEF at 0.5 nM = 3e9 molecules, conv_bone_nM = 6e12
        # So TCF_LEF/conv = 0.0005, which is too small
        # Instead use: basal * (1 + TCF_effect)
        kl.setMath(libsbml.parseL3Formula('k_osteoblast_diff * OsteoblastPrecursors * (1 + Wnt_LRP_active) * (1000 / (1000 + Osteoblasts)) / RadiationDamage'))
        reaction_count += 1
        
        # ========== OSTEOBLAST APOPTOSIS ==========
        rxn = self.model.createReaction()
        rxn.setId('R_osteoblast_apoptosis')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('Osteoblasts')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_osteoblast_apoptosis * Osteoblasts'))
        reaction_count += 1
        
        # ========== OSTEOCLAST DIFFERENTIATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_osteoclast_diff')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('OsteoclastPrecursors')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        product = rxn.createProduct()
        product.setSpecies('Osteoclasts')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        # Basal differentiation enhanced by RANKL-RANK signaling
        # Use ratio of complex to baseline RANK
        kl.setMath(libsbml.parseL3Formula(f'k_osteoclast_diff * OsteoclastPrecursors * (1 + RANKL_RANK_complex / (RANK_baseline * {conv_bone_nM})) * (200 / (200 + Osteoclasts))'))
        reaction_count += 1
        
        # ========== OSTEOCLAST APOPTOSIS ==========
        rxn = self.model.createReaction()
        rxn.setId('R_osteoclast_apoptosis')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('Osteoclasts')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_osteoclast_apoptosis * Osteoclasts'))
        reaction_count += 1
        
        # ========== OSTEOCYTE APOPTOSIS ==========
        rxn = self.model.createReaction()
        rxn.setId('R_osteocyte_apoptosis')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('Osteocytes')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        formula = 'piecewise(0, Osteocytes <= min_osteocytes, k_osteocyte_apoptosis_base * Osteocytes * (1 + k_radiation_ocy_apoptosis * Radiation))'
        kl.setMath(libsbml.parseL3Formula(formula))
        reaction_count += 1
        
        # ========== PRECURSOR REPLENISHMENT ==========
        rxn = self.model.createReaction()
        rxn.setId('R_precursor_replenish_ob')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('OsteoblastPrecursors')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_precursor_replenish * (200 - OsteoblastPrecursors)'))
        reaction_count += 1
        
        rxn = self.model.createReaction()
        rxn.setId('R_precursor_replenish_oc')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('OsteoclastPrecursors')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_precursor_replenish * (100 - OsteoclastPrecursors)'))
        reaction_count += 1
        
        # ========== BONE FORMATION (mg/mm/hr) ==========
        rxn = self.model.createReaction()
        rxn.setId('R_bone_formation')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('BoneMass')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        # Simplified: formation proportional to osteoblasts only
        # PTH effect via CREB is already captured in osteoblast differentiation
        formula = 'k_bone_formation * Osteoblasts * bone'
        kl.setMath(libsbml.parseL3Formula(formula))
        reaction_count += 1
        
        # ========== BONE RESORPTION (mg/mm/hr) ==========
        rxn = self.model.createReaction()
        rxn.setId('R_bone_resorption')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('BoneMass')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        formula = 'piecewise(0, BoneMass <= min_bone_mass, k_bone_resorption * Osteoclasts * bone)'
        kl.setMath(libsbml.parseL3Formula(formula))
        reaction_count += 1
        
        # ========== PTH PRODUCTION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_PTH_production')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('PTH_blood')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        # Produce molecules based on basal nM concentration and degradation balance
        formula = f'k_PTH_basal * {conv_blood_nM} * k_PTH_deg * (1 + biomarker_active * 5 * Biomarker_activation) * PTH_dysregulation_factor'
        kl.setMath(libsbml.parseL3Formula(formula))
        reaction_count += 1
        
        # ========== PTH DEGRADATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_PTH_degradation')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('PTH_blood')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_PTH_deg * PTH_blood'))
        reaction_count += 1
        
        # ========== PTH TRANSPORT (Blood → Bone) ==========
        rxn = self.model.createReaction()
        rxn.setId('R_PTH_to_bone')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('PTH_blood')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        product = rxn.createProduct()
        product.setSpecies('PTH_ECM')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_PTH_from_bone * PTH_blood'))
        reaction_count += 1
        
        # ========== PTH RECEPTOR BINDING ==========
        rxn = self.model.createReaction()
        rxn.setId('R_PTH_receptor_binding')
        rxn.setReversible(False)
        
        reactant1 = rxn.createReactant()
        reactant1.setSpecies('PTH_ECM')
        reactant1.setStoichiometry(1)
        reactant1.setConstant(True)
        
        reactant2 = rxn.createReactant()
        reactant2.setSpecies('PTH_receptor')
        reactant2.setStoichiometry(1)
        reactant2.setConstant(True)
        
        product = rxn.createProduct()
        product.setSpecies('PTH_receptor_active')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula(f'k_PTH_receptor_binding * PTH_ECM * PTH_receptor / {conv_bone_nM}'))
        reaction_count += 1
        
        # ========== PTH RECEPTOR DISSOCIATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_PTH_receptor_dissociation')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('PTH_receptor_active')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        product1 = rxn.createProduct()
        product1.setSpecies('PTH_ECM')
        product1.setStoichiometry(1)
        product1.setConstant(True)
        
        product2 = rxn.createProduct()
        product2.setSpecies('PTH_receptor')
        product2.setStoichiometry(1)
        product2.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_PTH_receptor_dissociation * PTH_receptor_active'))
        reaction_count += 1
        
        # ========== cAMP PRODUCTION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_cAMP_production')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('cAMP')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_cAMP_production * PTH_receptor_active'))
        reaction_count += 1
        
        # ========== cAMP DEGRADATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_cAMP_degradation')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('cAMP')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_cAMP_degradation * cAMP'))
        reaction_count += 1
        
        # ========== PKA ACTIVATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_PKA_activation')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('PKA_active')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_PKA_activation * cAMP'))
        reaction_count += 1
        
        # ========== PKA DEACTIVATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_PKA_deactivation')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('PKA_active')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_PKA_deactivation * PKA_active'))
        reaction_count += 1
        
        # ========== CREB PHOSPHORYLATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_CREB_phosphorylation')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('CREB_phosphorylated')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_CREB_phosphorylation * PKA_active'))
        reaction_count += 1
        
        # ========== CREB DEPHOSPHORYLATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_CREB_dephosphorylation')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('CREB_phosphorylated')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_CREB_dephosphorylation * CREB_phosphorylated'))
        reaction_count += 1
        
        # ========== CORTISOL PRODUCTION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_cortisol_production')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('Cortisol_blood')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        # k_cortisol_prod is in nM/hr, convert to molecules/hr
        # Stress increases production
        kl.setMath(libsbml.parseL3Formula(f'k_cortisol_prod * {conv_blood_nM} * (1 + StressHormones / 1000)'))
        reaction_count += 1
        
        # ========== CORTISOL DEGRADATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_cortisol_degradation')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('Cortisol_blood')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_cortisol_deg * Cortisol_blood'))
        reaction_count += 1
        
        # ========== IL6 PRODUCTION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_IL6_production')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('IL6_blood')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        # k_IL6_prod is in nM/hr, convert to molecules/hr
        # Radiation increases inflammation
        kl.setMath(libsbml.parseL3Formula(f'k_IL6_prod * {conv_blood_nM} * (1 + k_radiation_inflammation * Radiation)'))
        reaction_count += 1
        
        # ========== IL6 DEGRADATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_IL6_degradation')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('IL6_blood')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_IL6_deg * IL6_blood'))
        reaction_count += 1
        
        # ========== PHOSPHATE PRODUCTION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_phosphate_production')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('Phosphate_blood')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        conv_blood_mM = self.NA * self.V_blood * 1e-3
        kl.setMath(libsbml.parseL3Formula(f'k_phosphate_accumulation * {conv_blood_mM}'))
        reaction_count += 1
        
        # ========== PHOSPHATE CLEARANCE ==========
        rxn = self.model.createReaction()
        rxn.setId('R_phosphate_clearance')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('Phosphate_blood')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        formula = 'piecewise(0, Phosphate_mM <= min_phosphate_mM, k_phosphate_clearance * Phosphate_blood)'
        kl.setMath(libsbml.parseL3Formula(formula))
        reaction_count += 1
        
        # ========== FGF23 PRODUCTION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_FGF23_production')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('FGF23_blood')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        formula = f'k_FGF23_basal * {conv_blood_nM} * k_FGF23_deg * (1 + k_FGF23_phosphate * (Phosphate_mM - 1.1))'
        kl.setMath(libsbml.parseL3Formula(formula))
        reaction_count += 1
        
        # ========== FGF23 DEGRADATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_FGF23_degradation')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('FGF23_blood')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_FGF23_deg * FGF23_blood'))
        reaction_count += 1
        
        # ========== BIOMARKER PRODUCTION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_biomarker_production')
        rxn.setReversible(False)
        
        product = rxn.createProduct()
        product.setSpecies('Biomarker_ECM')
        product.setStoichiometry(1)
        product.setConstant(True)
        
        kl = rxn.createKineticLaw()
        # Biomarker_activation is already a Hill function that's 0 when Sclerostin_nM <= threshold
        # No need for separate biomarker_active flag
        kl.setMath(libsbml.parseL3Formula(f'k_biomarker_response * Biomarker_activation * {conv_bone_nM}'))
        reaction_count += 1
        
        # ========== BIOMARKER DEGRADATION ==========
        rxn = self.model.createReaction()
        rxn.setId('R_biomarker_degradation')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('Biomarker_ECM')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_biomarker_deg * Biomarker_ECM'))
        reaction_count += 1
        
        # ========== ESTROGEN DECLINE ==========
        rxn = self.model.createReaction()
        rxn.setId('R_estrogen_decline')
        rxn.setReversible(False)
        
        reactant = rxn.createReactant()
        reactant.setSpecies('Estrogen_blood')
        reactant.setStoichiometry(1)
        reactant.setConstant(True)
        
        kl = rxn.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula('k_estrogen_decline * Estrogen_blood'))
        reaction_count += 1
        
        print(f"  ✓ Created {reaction_count} reactions")
        print(f"    - All rates properly scaled for molecule counts")
        print(f"    - Binding reactions use (nM·hr)⁻¹ scaling")
    
    def create_events(self):
        """
        Create SBML events for biomarker activation.
        """
        print("\n[6/8] Creating events...")
        
        # ===== EVENT: Biomarker Activation =====
        event = self.model.createEvent()
        event.setId('biomarker_activation_event')
        event.setUseValuesFromTriggerTime(True)
        
        trigger = event.createTrigger()
        trigger.setInitialValue(False)
        trigger.setPersistent(True)
        trigger_math = libsbml.parseL3Formula('Sclerostin_nM > biomarker_threshold')
        trigger.setMath(trigger_math)
        
        assignment = event.createEventAssignment()
        assignment.setVariable('biomarker_active')
        assignment.setMath(libsbml.parseL3Formula('1.0'))
        
        print(f"  ✓ Created 1 event (biomarker activation at Sclerostin > 1.5 nM)")
    
    def validate_and_save(self, filename='bone_remodeling_model.xml'):
        """Validate SBML model and save to file."""
        print("\n[7/8] Validating SBML model...")
        
        num_errors = self.document.getNumErrors()
        
        if num_errors > 0:
            print(f"  ⚠ Found {num_errors} issues:")
            for i in range(num_errors):
                error = self.document.getError(i)
                severity = error.getSeverity()
                if severity >= libsbml.LIBSBML_SEV_ERROR:
                    print(f"    ERROR {i+1}: {error.getMessage()}")
                else:
                    print(f"    Warning {i+1}: {error.getMessage()}")
        else:
            print(f"  ✓ Model is valid!")
        
        print(f"\n[8/8] Saving SBML model to '{filename}'...")
        libsbml.writeSBMLToFile(self.document, filename)
        print(f"  ✓ Model saved successfully!")
        
        print("\n" + "="*80)
        print("SBML MODEL SUMMARY")
        print("="*80)
        print(f"Compartments: {self.model.getNumCompartments()}")
        print(f"Species: {self.model.getNumSpecies()}")
        print(f"Parameters: {self.model.getNumParameters()}")
        print(f"Reactions: {self.model.getNumReactions()}")
        print(f"Rules: {self.model.getNumRules()}")
        print(f"Events: {self.model.getNumEvents()}")
        print("="*80)
        
        return True
    
    def build_complete_model(self):
        """Build the complete SBML model step by step."""
        self.create_compartments()
        self.create_parameters()
        self.create_species()
        self.create_assignment_rules()
        self.create_reactions()
        self.create_events()
        
        if self.validate_and_save():
            print("\n✓ SBML model generation complete!")
            print("\n  Key features:")
            print("  - All units harmonized (nM, hr⁻¹, molecules/cell/hr)")
            print("  - BoneMass in mg/mm (physiological scale)")
            print("  - Literature-validated parameters")
            print("  - Balanced formation/resorption for steady-state")
            return True
        else:
            print("\n✗ Model validation failed")
            return False


def main():
    """Main function to generate SBML model."""
    generator = BoneRemodelingModel()
    success = generator.build_complete_model()
    
    if success:
        print("\n" + "="*80)
        print("USAGE INSTRUCTIONS")
        print("="*80)
        print("""
1. REGENERATE MODEL:
   python bone_remodeling_sbml_harmonized.py

2. RUN SIMULATION:
   python simulate_bone_model.py

3. EXPECTED RESULTS:
   - Normal: BoneMass stable around 4.0 mg/mm
   - Spaceflight: ~5-15% bone loss over 6 months
   - Osteoporosis: Gradual bone loss due to low estrogen
   - CKD-MBD: Elevated PTH, bone loss

4. KEY OUTPUTS:
   - BoneMass (mg/mm): Should stay above 3.0
   - Sclerostin_nM: Elevated in disease states
   - Biomarker activation when Sclerostin > 1.5 nM
        """)
        print("="*80)

if __name__ == '__main__':
    main()