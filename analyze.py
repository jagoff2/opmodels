
import onnx
import onnxruntime as ort
from onnx import shape_inference, checker
import argparse, json

def extract_metadata(onnx_path):
    model = onnx.load(onnx_path)
    checker.check_model(model)
    inferred = shape_inference.infer_shapes(model)
    checker.check_model(inferred)

    meta = {
        "ir_version": inferred.ir_version,
        "producer_name": inferred.producer_name,
        "inputs": [],
        "outputs": [],
        "initializers": [],
        "nodes": []
    }

    # Inputs
    for inp in inferred.graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value else None)
        meta["inputs"].append({
            "name": inp.name,
            "dtype": inp.type.tensor_type.elem_type,
            "shape": shape
        })

    # Outputs
    for out in inferred.graph.output:
        shape = []
        for d in out.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value else None)
        meta["outputs"].append({
            "name": out.name,
            "dtype": out.type.tensor_type.elem_type,
            "shape": shape
        })

    # Initializers (parameters)
    total_params = 0
    for init in inferred.graph.initializer:
        dims = list(init.dims)
        count = 1
        for s in dims: count *= s
        total_params += count
        meta["initializers"].append({
            "name": init.name,
            "dtype": init.data_type,
            "dims": dims,
            "count": count
        })
    meta["total_parameters"] = total_params

    # Nodes
    for node in inferred.graph.node:
        meta["nodes"].append({
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output)
        })

    return meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract ONNX model metadata"
    )
    parser.add_argument("model_path", help="Path to the ONNX model file")
    parser.add_argument("--out", default="metadata.json",
                        help="Output JSON filename")
    args = parser.parse_args()

    data = extract_metadata(args.model_path)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Metadata written to {args.out}")
