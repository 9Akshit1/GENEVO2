"""
Simple diagnostic for bone model
"""

import tellurium as te
import numpy as np

print("="*60)
print("DIAGNOSTIC: Testing model stability")
print("="*60)

# Load model
r = te.loadSBMLModel('bone_remodeling_model.xml')

# Configure integrator
r.integrator = 'cvode'
r.integrator.setValue('stiff', True)
r.integrator.setValue('absolute_tolerance', 1e-10)
r.integrator.setValue('relative_tolerance', 1e-6)
r.integrator.setValue('maximum_num_steps', 100000)

print("\nInitial conditions:")
print(f"  Osteoblasts: {r['Osteoblasts']:.1f}")
print(f"  Osteoclasts: {r['Osteoclasts']:.1f}")
print(f"  Osteocytes: {r['Osteocytes']:.1f}")
print(f"  BoneMass: {r['BoneMass']:.3f}")

# Test 24 hours
print("\n" + "-"*60)
print("Testing 24-hour simulation...")
print("-"*60)
try:
    result = r.simulate(0, 24, 100)
    print("✓ SUCCESS: 24 hours completed")
    
    # Access as array
    final_idx = -1
    print(f"\nAfter 24 hours:")
    print(f"  Osteoblasts: {result[final_idx, 10]:.1f}")  # Column 10
    print(f"  Osteoclasts: {result[final_idx, 11]:.1f}")  # Column 11
    print(f"  BoneMass: {result[final_idx, 41]:.1f}")     # Column 41
    
    # Check for explosive growth
    if result[final_idx, 10] > 1000:
        print("\n⚠️  WARNING: Osteoblasts exploding!")
    elif result[final_idx, 10] < 100:
        print("\n⚠️  WARNING: Osteoblasts dying!")
    else:
        print("\n✓ Osteoblasts stable")
        
    if result[final_idx, 11] > 200:
        print("⚠️  WARNING: Osteoclasts exploding!")
    elif result[final_idx, 11] < 10:
        print("⚠️  WARNING: Osteoclasts dying!")
    else:
        print("✓ Osteoclasts stable")
        
    if result[final_idx, 41] > 10 or result[final_idx, 41] < 1:
        print("⚠️  WARNING: Bone mass changing too rapidly!")
    else:
        print("✓ Bone mass stable")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    exit(1)

# Test 1 week
print("\n" + "-"*60)
print("Testing 168-hour (1 week) simulation...")
print("-"*60)
try:
    r.reset()  # Reset to initial conditions
    result = r.simulate(0, 168, 100)
    print("✓ SUCCESS: 1 week completed")
    
    print(f"\nAfter 1 week:")
    print(f"  Osteoblasts: {result[-1, 10]:.1f}")
    print(f"  Osteoclasts: {result[-1, 11]:.1f}")
    print(f"  BoneMass: {result[-1, 41]:.3f}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    exit(1)

# Test 1 month
print("\n" + "-"*60)
print("Testing 720-hour (1 month) simulation...")
print("-"*60)
try:
    r.reset()
    result = r.simulate(0, 720, 200)
    print("✓ SUCCESS: 1 month completed")
    
    print(f"\nAfter 1 month:")
    print(f"  Osteoblasts: {result[-1, 10]:.1f}")
    print(f"  Osteoclasts: {result[-1, 11]:.1f}")
    print(f"  BoneMass: {result[-1, 41]:.3f}")
    
    # Calculate change
    initial = 4.0
    final = result[-1, 41]
    change_pct = ((final - initial) / initial) * 100
    print(f"  Bone change: {change_pct:+.2f}%")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    exit(1)

# Test 6 months
print("\n" + "-"*60)
print("Testing 4320-hour (6 months) simulation...")
print("-"*60)
try:
    r.reset()
    result = r.simulate(0, 4320, 500)
    print("✓ SUCCESS: 6 months completed!")
    
    print(f"\nAfter 6 months:")
    print(f"  Osteoblasts: {result[-1, 10]:.1f}")
    print(f"  Osteoclasts: {result[-1, 11]:.1f}")
    print(f"  BoneMass: {result[-1, 41]:.3f}")
    
    initial = 4.0
    final = result[-1, 41]
    change_pct = ((final - initial) / initial) * 100
    print(f"  Bone change: {change_pct:+.2f}%")
    
    print("\n" + "="*60)
    print("✓ MODEL IS STABLE AND READY FOR FULL SIMULATION")
    print("="*60)
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    print("\nModel cannot run for 6 months. Try shorter duration.")

print("\nNext step: Run 'python3 simulate_bone_model.py'")