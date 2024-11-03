import onnxruntime as ort

def check_cuda_availability():
    try:
        # Check available providers
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        if 'CUDAExecutionProvider' in providers:
            # Try to create a simple session with CUDA
            options = ort.SessionOptions()
            dummy_model_path = "models/2d106det.onnx"  # You'll need a valid ONNX model for real testing
            print(f"Trying to create an InferenceSession with CUDAExecutionProvider for model: {dummy_model_path}")
            _ = ort.InferenceSession(dummy_model_path, providers=['CUDAExecutionProvider'])
            print("CUDAExecutionProvider is available and working.")
            return True
        print("CUDAExecutionProvider is not available.")
        return False
    except Exception as e:
        print(f"CUDA check error: {str(e)}")
        return False

if __name__ == "__main__":
    check_cuda_availability()
    