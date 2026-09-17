"""Prepare the UVA packet-level dataset for the common hourly benchmark."""

import argparse
import json
from src.dataset_standardization import prepare_uva_gateway_a


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize UVA LoRaWAN data")
    parser.add_argument("--source-dir", default="dataset")
    parser.add_argument("--output-file", default="data/experiment_2_uva_gatewayA_hourly.csv")
    args = parser.parse_args()
    print(json.dumps(prepare_uva_gateway_a(args.source_dir, args.output_file), indent=2))


if __name__ == "__main__":
    main()
