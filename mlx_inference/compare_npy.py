import argparse
import mlx.core as mx
import numpy as np

parser = argparse.ArgumentParser(
    description="Compare two .npy files for numerical similarity"
)
parser.add_argument("file1", type=str, help="Path to first .npy file")
parser.add_argument("file2", type=str, help="Path to second .npy file")


def main():
    args = parser.parse_args()

    arr1_np = np.load(args.file1)
    arr2_np = np.load(args.file2)

    arr1 = mx.array(arr1_np)
    arr2 = mx.array(arr2_np)

    if arr1.shape != arr2.shape:
        print(f"Shape mismatch: {arr1.shape} vs {arr2.shape}")
        return

    # Check numerical similarity
    is_close = mx.allclose(arr1, arr2, rtol=1e-3, atol=1e-3)
    mx.eval(is_close)
    max_diff = mx.abs(arr1 - arr2).max()
    if is_close:
        print("Arrays match within tolerance")
    else:
        # If they don't match, might be helpful to see the max difference
        print("Arrays differ.")
    print(f"Max absolute difference: {max_diff}")


if __name__ == "__main__":
    main()
