import yaml
import argparse

def create_yaml_file(output_path, dataset_path):
    data = {
        "path": dataset_path,
        "train": "images/train/",
        "val": "images/val/",
        "names": {0: "Bleeding"}
    }
    
    yaml_file = f"{output_path}/custom_data1.yaml"
    with open(yaml_file, "w") as file:
        yaml.dump(data, file, default_flow_style=False)
    
    print(f"YAML file saved at: {yaml_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the YAML file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Base dataset path")
    
    args = parser.parse_args()
    create_yaml_file(args.output_path, args.dataset_path)
