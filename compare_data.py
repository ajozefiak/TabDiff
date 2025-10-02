import argparse
import pandas as pd
import json
from sdmetrics.reports.single_table import QualityReport

def load_metadata(info_path):
    with open(info_path, 'r') as f:
        info = json.load(f)
    metadata = info['metadata']
    # Ensure keys are integers for SDMetrics
    metadata['columns'] = {int(k): v for k, v in metadata['columns'].items()}
    return metadata

def main():
    parser = argparse.ArgumentParser(description="Compare two tabular datasets using shape and trend metrics.")
    parser.add_argument('--data1', type=str, required=True, help='Path to first dataset CSV')
    parser.add_argument('--data2', type=str, required=True, help='Path to second dataset CSV')
    parser.add_argument('--info', type=str, required=True, help='Path to info.json containing metadata')
    args = parser.parse_args()

    df1 = pd.read_csv(args.data1)
    df2 = pd.read_csv(args.data2)

    # Align columns for SDMetrics
    df1.columns = range(len(df1.columns))
    df2.columns = range(len(df2.columns))

    metadata = load_metadata(args.info)

    report = QualityReport()
    report.generate(df1, df2, metadata)
    quality = report.get_properties()
    shape = quality['Score'][0]
    trend = quality['Score'][1]
    print(f"Shape similarity: {shape:.4f}")
    print(f"Trend similarity: {trend:.4f}")

    # Optionally, print details
    shape_details = report.get_details(property_name='Column Shapes')
    trend_details = report.get_details(property_name='Column Pair Trends')
    print("\nShape details:\n", shape_details)
    print("\nTrend details:\n", trend_details)

if __name__ == "__main__":
    main()

# Usage Example
# python compare_data.py --data1 synthetic/adult/real.csv --data2 synthetic/adult/samples.csv --info data/adult/info.json