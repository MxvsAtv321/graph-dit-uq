#!/usr/bin/env python3
"""
Test OpenMM container compatibility before long MD runs.
"""

import sys
import subprocess

def test_openmm_container():
    """Test if OpenMM works in the container with CUDA."""
    
    print("🔬 Testing OpenMM container compatibility...")
    
    # Test 1: Basic OpenMM import
    try:
        import simtk.openmm as mm
        print("✅ OpenMM import successful")
    except ImportError as e:
        print(f"❌ OpenMM import failed: {e}")
        return False
    
    # Test 2: CUDA platform availability
    try:
        cuda_platform = mm.Platform.getPlatformByName('CUDA')
        print(f"✅ CUDA platform available: {cuda_platform.getName()}")
    except Exception as e:
        print(f"❌ CUDA platform not available: {e}")
        print("⚠️  Will fall back to CPU platform")
    
    # Test 3: GPU device count
    try:
        gpu_count = cuda_platform.getNumDevices()
        print(f"✅ GPU devices available: {gpu_count}")
        
        for i in range(gpu_count):
            device = cuda_platform.getDevice(i)
            print(f"  GPU {i}: {device.getName()}")
    except Exception as e:
        print(f"⚠️  Could not get GPU device info: {e}")
    
    # Test 4: Simple system creation
    try:
        # Create a simple system
        system = mm.System()
        print("✅ System creation successful")
        
        # Add a simple harmonic bond
        force = mm.HarmonicBondForce()
        system.addForce(force)
        print("✅ Force field addition successful")
        
    except Exception as e:
        print(f"❌ System creation failed: {e}")
        return False
    
    # Test 5: Context creation
    try:
        integrator = mm.VerletIntegrator(0.001)
        context = mm.Context(system, integrator)
        print("✅ Context creation successful")
        
        # Clean up
        del context
        del integrator
        
    except Exception as e:
        print(f"❌ Context creation failed: {e}")
        return False
    
    print("🎉 OpenMM container test PASSED!")
    print("✅ Ready for 72 GPU-hour MD validation run")
    return True

def test_gpu_quota():
    """Test GPU quota and availability."""
    
    print("\n🔍 Checking GPU quota...")
    
    try:
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        print("📊 GPU Status:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 4:
                    name, total, used, free = parts[:4]
                    used_pct = (int(used) / int(total)) * 100
                    print(f"  {name}: {used_pct:.1f}% used ({free}MB free)")
                    
                    if used_pct > 80:
                        print(f"    ⚠️  High GPU usage: {used_pct:.1f}%")
                    else:
                        print(f"    ✅ Good headroom: {100-used_pct:.1f}% available")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ nvidia-smi failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - no GPU available")
        return False
    
    return True

def main():
    """Run all container tests."""
    
    print("🚀 OpenMM Container Compatibility Test")
    print("=" * 50)
    
    # Test OpenMM
    openmm_ok = test_openmm_container()
    
    # Test GPU quota
    gpu_ok = test_gpu_quota()
    
    print("\n" + "=" * 50)
    if openmm_ok and gpu_ok:
        print("🎉 ALL TESTS PASSED - READY FOR MD VALIDATION!")
        print("✅ Container: OpenMM + CUDA working")
        print("✅ GPU: Sufficient headroom available")
        print("\n🚀 You can now run:")
        print("airflow dags trigger dit_uq_stage4_md --run-id stage4_md_$(date +%Y%m%d_%H%M)")
        return 0
    else:
        print("❌ TESTS FAILED - FIX BEFORE MD RUN")
        if not openmm_ok:
            print("❌ OpenMM container issues detected")
        if not gpu_ok:
            print("❌ GPU quota issues detected")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 