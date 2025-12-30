"""
Comprehensive Pipeline Verification Script
Tests all components of the multimodal AI pipeline
"""

import sys
import json
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def test_imports():
    """Test all required imports"""
    print_header("1. Testing Python Imports")
    
    tests = [
        ("streamlit", "Streamlit Web Framework"),
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("diffusers", "Diffusers Library"),
        ("transformers", "Transformers Library"),
        ("ollama", "Ollama API"),
        ("tiktoken", "TikToken"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
    ]
    
    success_count = 0
    for module, name in tests:
        try:
            __import__(module)
            print(f"  ✓ {name:30} ({module})")
            success_count += 1
        except ImportError as e:
            print(f"  ✗ {name:30} ({module})")
            print(f"    Error: {e}")
    
    print(f"\n  Result: {success_count}/{len(tests)} imports successful")
    return success_count == len(tests)

def test_local_modules():
    """Test local pipeline modules"""
    print_header("2. Testing Local Pipeline Modules")
    
    modules = [
        ("prompt_enhancement_module", "PromptEnhancer"),
        ("image_generation_module", "ImageGenerator"),
        ("annotation_module", "AnnotationEngine"),
    ]
    
    success_count = 0
    for module_name, class_name in modules:
        try:
            module = __import__(module_name)
            cls = getattr(module, class_name)
            print(f"  ✓ {class_name:20} from {module_name}")
            success_count += 1
        except (ImportError, AttributeError) as e:
            print(f"  ✗ {class_name:20} from {module_name}")
            print(f"    Error: {e}")
    
    print(f"\n  Result: {success_count}/{len(modules)} modules loaded")
    return success_count == len(modules)

def test_ollama_connection():
    """Test Ollama service"""
    print_header("3. Testing Ollama Service")
    
    try:
        import ollama
        print("  Checking Ollama connection on localhost:11434...")
        
        response = ollama.generate(
            model="llama2:latest",
            prompt="Hello",
            stream=False,
            options={"num_predict": 5}
        )
        
        if response and response.get("response"):
            print(f"  ✓ Ollama service is running")
            print(f"    Model: llama2:latest")
            print(f"    Response received: {response.get('response')[:50]}...")
            return True
        else:
            print("  ✗ No response from Ollama")
            return False
    except Exception as e:
        print(f"  ✗ Ollama connection failed: {e}")
        return False

def test_prompt_enhancement():
    """Test prompt enhancement"""
    print_header("4. Testing Prompt Enhancement")
    
    try:
        from prompt_enhancement_module import PromptEnhancer
        
        print("  Initializing PromptEnhancer...")
        enhancer = PromptEnhancer()
        
        test_prompt = "A robotic arm picking up a cardboard box"
        print(f"  Test prompt: '{test_prompt}'")
        print("  Enhancing for Flux model...")
        
        result = enhancer.enhance_prompt(test_prompt, "flux")
        
        if result and result.get("enhanced_json"):
            enhanced = result["enhanced_json"]
            token_count = result.get("token_count", 0)
            
            print(f"  ✓ Enhancement successful!")
            print(f"    Token count: {token_count}")
            print(f"    Summary: {enhanced.get('Summary', 'N/A')[:60]}...")
            print(f"    Objects: {enhanced.get('Attributes', {}).get('Object Class', 'N/A')}")
            return True
        else:
            print("  ✗ Enhancement failed")
            return False
    except Exception as e:
        print(f"  ✗ Enhancement error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_paths():
    """Test if model paths exist"""
    print_header("5. Testing Model Paths")
    
    paths_to_check = [
        ("/mnt/myssd/models/flux/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44", "Flux Model"),
        ("/home/robot/Documents/VM_Annotation_Pipeline/Grounded-SAM-2", "Grounding DINO"),
    ]
    
    success_count = 0
    for path, name in paths_to_check:
        if Path(path).exists():
            print(f"  ✓ {name:20} ✓ Found")
            success_count += 1
        else:
            print(f"  ✗ {name:20} ✗ Not found at {path}")
    
    print(f"\n  Result: {success_count}/{len(paths_to_check)} models found")
    return success_count > 0

def test_output_directories():
    """Test output directories"""
    print_header("6. Testing Output Directories")
    
    import os
    
    directories = [
        "pipeline_outputs",
        "pipeline_outputs/generated_images",
        "pipeline_outputs/annotations",
    ]
    
    success_count = 0
    for dirname in directories:
        try:
            os.makedirs(dirname, exist_ok=True)
            print(f"  ✓ {dirname:40} ✓ Ready")
            success_count += 1
        except Exception as e:
            print(f"  ✗ {dirname:40} ✗ Error: {e}")
    
    print(f"\n  Result: {success_count}/{len(directories)} directories ready")
    return success_count == len(directories)

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("  🤖 MULTIMODAL AI PIPELINE - VERIFICATION TEST")
    print("="*80)
    
    tests = [
        ("Python Imports", test_imports),
        ("Local Modules", test_local_modules),
        ("Ollama Service", test_ollama_connection),
        ("Prompt Enhancement", test_prompt_enhancement),
        ("Model Paths", test_model_paths),
        ("Output Directories", test_output_directories),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_header("VERIFICATION SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:30} {status}")
    
    print(f"\n  Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED! Pipeline is ready to use.")
        print("\n  To start the pipeline:")
        print("    streamlit run app.py")
        return 0
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed. Please check above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
