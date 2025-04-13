import os

import numpy as np
import tensorflow as tf

from src.training.Generator import Generator


def validate_checkpoint_changes(checkpoint_path1, checkpoint_path2, test_image=None):
    """
    Validates that two checkpoints have different weights and produce different outputs.

    Args:
        checkpoint_path1: Path to first checkpoint
        checkpoint_path2: Path to second checkpoint
        test_image: Optional test image to compare generations

    Returns:
        dict: Dictionary with validation results
    """
    print(
        f"Validating changes between checkpoints:\n- {checkpoint_path1}\n- {checkpoint_path2}"
    )

    # Create temporary models for validation
    temp_generator1 = Generator()
    temp_generator2 = Generator()

    # Create temporary checkpoints
    temp_checkpoint1 = tf.train.Checkpoint(generator=temp_generator1)
    temp_checkpoint2 = tf.train.Checkpoint(generator=temp_generator2)

    # Restore from checkpoints
    status1 = temp_checkpoint1.restore(checkpoint_path1)
    status2 = temp_checkpoint2.restore(checkpoint_path2)

    try:
        status1.assert_existing_objects_matched()
        status2.assert_existing_objects_matched()
    except Exception as e:
        return {"error": f"Failed to restore checkpoints: {e}"}

    # Compare a few weights to detect differences
    results = {"weights_differ": False, "output_differs": False}

    # Get weights from first layer for comparison
    weights1 = temp_generator1.layers[1].weights[0].numpy()
    weights2 = temp_generator2.layers[1].weights[0].numpy()

    # Check if weights are identical
    weight_diff = np.mean(np.abs(weights1 - weights2))
    results["weights_differ"] = weight_diff > 0
    results["weight_diff_magnitude"] = float(weight_diff)

    # If provided with a test image, compare outputs
    if test_image is not None:
        out1 = temp_generator1(test_image, training=False)
        out2 = temp_generator2(test_image, training=False)

        output_diff = np.mean(np.abs(out1.numpy() - out2.numpy()))
        results["output_differs"] = output_diff > 0
        results["output_diff_magnitude"] = float(output_diff)

    # Log results
    for key, value in results.items():
        print(f"- {key}: {value}")

    return results


def compare_model_params(model1, model2):
    """
    Compare two models layer by layer to find differences.

    Args:
        model1: First TensorFlow model
        model2: Second TensorFlow model

    Returns:
        dict: Dictionary with detailed layer-by-layer comparison
    """
    results = {"layer_diffs": []}

    # Ensure both models have the same number of layers
    if len(model1.layers) != len(model2.layers):
        results["error"] = (
            f"Layer count mismatch: {len(model1.layers)} vs {len(model2.layers)}"
        )
        return results

    total_params = 0
    diff_params = 0

    # Compare layer by layer
    for i, (layer1, layer2) in enumerate(zip(model1.layers, model2.layers)):
        layer_result = {
            "layer_name": layer1.name,
            "layer_index": i,
            "weights_differ": False,
            "diff_magnitude": 0.0,
        }

        # Skip layers without weights
        if not layer1.weights:
            continue

        # Compare weights for each layer
        for w1, w2 in zip(layer1.weights, layer2.weights):
            w1_np = w1.numpy()
            w2_np = w2.numpy()

            # Count parameters
            param_count = np.prod(w1_np.shape)
            total_params += param_count

            # Calculate difference
            diff = np.abs(w1_np - w2_np)
            mean_diff = float(np.mean(diff))
            max_diff = float(np.max(diff))

            # Count differing parameters
            diff_param_count = np.sum(diff > 0)
            diff_params += diff_param_count

            # Update layer result
            if mean_diff > 0:
                layer_result["weights_differ"] = True
                layer_result["diff_magnitude"] = max(
                    layer_result["diff_magnitude"], mean_diff
                )
                layer_result["diff_percent"] = (
                    float(diff_param_count) / param_count * 100
                )

        results["layer_diffs"].append(layer_result)

    # Summary statistics
    results["total_params"] = int(total_params)
    results["diff_params"] = int(diff_params)
    results["diff_percent"] = (
        float(diff_params) / total_params * 100 if total_params > 0 else 0
    )

    return results


def inspect_checkpoint_files(checkpoint_dir, latest_n=3):
    """
    Inspect checkpoint files to analyze their structure and size.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        latest_n: Number of latest checkpoints to analyze

    Returns:
        dict: Dictionary with checkpoint file information
    """
    import glob
    import os

    # Find checkpoint files
    index_files = glob.glob(os.path.join(checkpoint_dir, "*.index"))

    if not index_files:
        return {"error": f"No checkpoint files found in {checkpoint_dir}"}

    # Sort by modification time to get latest checkpoints
    index_files.sort(key=os.path.getmtime, reverse=True)
    index_files = index_files[:latest_n]

    results = {"checkpoints": []}

    for index_file in index_files:
        # Get base path without extension
        base_path = index_file[:-6]  # Remove .index
        data_files = glob.glob(f"{base_path}.data-*")

        checkpoint_info = {"path": base_path, "data_files": [], "total_size_bytes": 0}

        # Check .index file
        if os.path.exists(index_file):
            index_size = os.path.getsize(index_file)
            checkpoint_info["index_size_bytes"] = index_size
            checkpoint_info["total_size_bytes"] += index_size

        # Check all data files
        for data_file in data_files:
            if os.path.exists(data_file):
                data_size = os.path.getsize(data_file)
                checkpoint_info["data_files"].append(
                    {"path": data_file, "size_bytes": data_size}
                )
                checkpoint_info["total_size_bytes"] += data_size

        results["checkpoints"].append(checkpoint_info)

    # Add summary
    if results["checkpoints"]:
        # Check if all data files have the same size
        data_sizes = []
        for ckpt in results["checkpoints"]:
            for data_file in ckpt["data_files"]:
                data_sizes.append(data_file["size_bytes"])

        results["all_same_size"] = len(set(data_sizes)) == 1 if data_sizes else False
        results["data_file_size"] = data_sizes[0] if data_sizes else 0

    return results


def validate_checkpoint_directory(checkpoint_dir, test_dataset=None):
    """
    Comprehensive checkpoint directory validation.

    Args:
        checkpoint_dir: Directory containing checkpoints
        test_dataset: Optional test dataset for validating outputs

    Returns:
        dict: Dictionary with validation results
    """
    # Step 1: Inspect checkpoint files
    file_inspection = inspect_checkpoint_files(checkpoint_dir)

    # Step 2: Find available checkpoints
    checkpoints = []
    checkpoint_obj = tf.train.CheckpointManager(
        tf.train.Checkpoint(), checkpoint_dir, max_to_keep=100
    )

    for ckpt in checkpoint_obj.checkpoints:
        if tf.io.gfile.exists(f"{ckpt}.index"):
            checkpoints.append(ckpt)

    # Add best model if it exists
    best_model_path = os.path.join(checkpoint_dir, "best_model")
    if tf.io.gfile.exists(f"{best_model_path}.index"):
        checkpoints.append(best_model_path)

    results = {
        "file_inspection": file_inspection,
        "checkpoints_found": len(checkpoints),
        "checkpoint_paths": checkpoints,
    }

    # Step 3: Compare first and last checkpoint if we have multiple
    if len(checkpoints) >= 2:
        # Get a test image if possible
        test_image = None
        if test_dataset is not None:
            for example_input, _ in test_dataset.take(1):
                test_image = example_input
                break

        # Compare first and last checkpoint
        validation_results = validate_checkpoint_changes(
            checkpoints[0], checkpoints[-1], test_image
        )
        results["validation_results"] = validation_results

    return results
