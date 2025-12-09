"""
Quick diagnostic to identify unstable species in the bone model
"""

import tellurium as te
import numpy as np

print("="*60)
print("DIAGNOSTIC: Short-term model behavior")
print("="*60)

# Load model
r = te.loadSBMLModel('bone_remodeling_model.xml')

# Configure for short simulation
r.integrator = 'cvode'
r.integrator.setValue('stiff', True)
r.integrator.setValue('absolute_tolerance', 1e-10)
r.integrator.setValue('relative_tolerance', 1e-6)
r.integrator.setValue('maximum_num_steps', 100000)

print("\nInitial values:")
print(f"  Osteoblasts: {r['Osteoblasts']:.1f}")
print(f"  Osteoclasts: {r['Osteoclasts']:.1f}")
print(f"  Osteocytes: {r['Osteocytes']:.1f}")
print(f"  BoneMass: {r['BoneMass']:.3f}")

# Try very short simulation
print("\nAttempting 24-hour simulation...")
try:
    result = r.simulate(0, 24, 100)
    print("✓ 24 hours successful")
    
    print(f"\nAfter 24 hours:")
    # Get column indices
    col_names = list(result.colnames)
    ob_idx = col_names.index('Osteoblasts')
    oc_idx = col_names.index('Osteoclasts')
    ocy_idx = col_names.index('Osteocytes')
    bone_idx = col_names.index('BoneMass')
    
    ob_24h = result[-1, ob_idx]
    oc_24h = result[-1, oc_idx]
    ocy_24h = result[-1, ocy_idx]
    bone_24h = result[-1, bone_idx]
    
    print(f"  Osteoblasts: {ob_24h:.1f}")
    print(f"  Osteoclasts: {oc_24h:.1f}")
    print(f"  Osteocytes: {ocy_24h:.1f}")
    print(f"  BoneMass: {bone_24h:.3f}")
    
    # Check for explosive growth
    if ob_24h > 1000:
        print("\n⚠️  Osteoblasts are exploding!")
        print("  Problem: k_osteoblast_diff too high or k_osteoblast_apoptosis too low")
    
    if oc_24h > 200:
        print("\n⚠️  Osteoclasts are exploding!")
        print("  Problem: k_osteoclast_diff too high or k_osteoclast_apoptosis too low")
    
    if ocy_24h < 1000:
        print("\n⚠️  Osteocytes are dying too fast!")
        print("  Problem: k_osteocyte_apoptosis_base too high")
    
    if bone_24h > 10 or bone_24h < 1:
        print("\n⚠️  Bone mass changing too rapidly!")
        print("  Problem: k_bone_formation or k_bone_resorption too high")
    
    # Try 1 week
    print("\nAttempting 168-hour (1 week) simulation...")
    result = r.simulate(0, 168, 100)
    print("✓ 1 week successful")
    
    ob_week = result[-1, col_names.index('Osteoblasts')]
    oc_week = result[-1, col_names.index('Osteoclasts')]
    bone_week = result[-1, col_names.index('BoneMass')]
    print(f"\nAfter 1 week:")
    print(f"  Osteoblasts: {ob_week:.1f}")
    print(f"  Osteoclasts: {oc_week:.1f}")
    print(f"  BoneMass: {bone_week:.3f}")
    
    # Try 1 month
    print("\nAttempting 720-hour (1 month) simulation...")
    result = r.simulate(0, 720, 200)
    print("✓ 1 month successful!")
    
    ob_month = result[-1, col_names.index('Osteoblasts')]
    oc_month = result[-1, col_names.index('Osteoclasts')]
    bone_month = result[-1, col_names.index('BoneMass')]
    print(f"\nAfter 1 month:")
    print(f"  Osteoblasts: {ob_month:.1f}")
    print(f"  Osteoclasts: {oc_month:.1f}")
    print(f"  BoneMass: {bone_month:.3f}")
    
except Exception as e:
    print(f"❌ Simulation failed: {e}")
    print("\nThe model has severe parameter instability.")
    print("Likely causes:")
    print("  1. k_osteoblast_diff = 0.0003 is too high")
    print("  2. k_osteoclast_diff = 0.00015 is too high")
    print("  3. Wnt_LRP_active formula causing runaway positive feedback")
    print("\nRecommended fixes:")
    print("  - Reduce k_osteoblast_diff to 0.00001")
    print("  - Reduce k_osteoclast_diff to 0.000005")
    print("  - Add saturation to cell differentiation reactions")

print("\n" + "="*60)


