import json
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import pandas as pd


def save_to_file(data, filename):
    """Helper function to save data to a file."""
    with open(filename, 'w') as f:
        if filename.endswith('.json'):
            json.dump(data, f, indent=2)
        else:
            f.write(data)

def save_visualization(fig, filename):
    """Helper function to save a visualization to a file."""
    print(fig)  # Should print something like <Figure size 1400x800 with 1 Axes>
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def load_data(json_file_path):
    """Load JSON data into a Pandas DataFrame."""
    try:
        df = pd.read_json(json_file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return None
    except ValueError:
        print(f"Error: Invalid JSON format in '{json_file_path}'.")
        return None


def calculate_station_time(station_processes):
    return sum(p['expectedTimeInMin'] for p in station_processes)


def check_dependencies_met(process_id, allocated_processes, process_dependencies):
    return all(dep in allocated_processes for dep in process_dependencies.get(process_id, []))


def allocate_processes_to_stations(df, respect_dependencies=True, throughput_target=1.0):
    """
    Allocate processes to stations using Pandas for data manipulation.

    Args:
        df (pd.DataFrame): DataFrame containing process data.
        respect_dependencies (bool): Whether to respect process dependencies.
        throughput_target (float): Target throughput in minutes.

    Returns:
        dict: Station allocation details.
    """
    stations = []
    station_id = 1
    allocated_processes = set()

    if respect_dependencies:
        # Validate dependencies
        for process_id, deps in df.set_index('id')['dependency'].items():
            for dep in deps:
                if dep not in df['id'].values:
                    raise ValueError(f"Process '{process_id}' has invalid dependency: '{dep}'")

        while len(allocated_processes) < len(df):
            found_new_station = False
            for _, process in df.iterrows():
                if process['id'] in allocated_processes:
                    continue

                # Check if dependencies are met
                if all(dep in allocated_processes for dep in process['dependency']):
                    machine_type = process['machineType']
                    current_station = [process]
                    allocated_processes.add(process['id'])

                    # Add processes with the same machine type
                    for _, next_process in df.iterrows():
                        if next_process['id'] in allocated_processes:
                            continue
                        if (next_process['machineType'] == machine_type and
                            all(dep in allocated_processes for dep in next_process['dependency']) and
                            (sum(p['expectedTimeInMin'] for p in current_station)) + next_process['expectedTimeInMin'] <= throughput_target):
                            current_station.append(next_process)
                            allocated_processes.add(next_process['id'])

                    # Add station to the list
                    stations.append({
                        'station_id': f"S{station_id}",
                        'processes': [{'id': p['id'], 'name': p['name']} for p in current_station],
                        'total_expected_time': sum(p['expectedTimeInMin'] for p in current_station),
                        'waste_time': throughput_target - sum(p['expectedTimeInMin'] for p in current_station),
                        'machines_required': 1
                    })
                    station_id += 1
                    found_new_station = True
                    break

            if not found_new_station:
                break
    else:
        # Group processes by machine type
        machine_type_groups = df.groupby('machineType')

        for machine_type, group in machine_type_groups:
            station_processes = []
            total_expected_time = 0
            for _, process in group.iterrows():
                if total_expected_time + process['expectedTimeInMin'] <= throughput_target:
                    station_processes.append(process)
                    total_expected_time += process['expectedTimeInMin']
                else:
                    stations.append({
                        'station_id': f"S{station_id}",
                        'processes': [{'id': p['id'], 'name': p['name']} for p in station_processes],
                        'total_expected_time': total_expected_time,
                        'waste_time': throughput_target - total_expected_time,
                        'machines_required': math.ceil(total_expected_time)
                    })
                    station_id += 1
                    station_processes = [process]
                    total_expected_time = process['expectedTimeInMin']

            if station_processes:
                stations.append({
                    'station_id': f"S{station_id}",
                    'processes': [{'id': p['id'], 'name': p['name']} for p in station_processes],
                    'total_expected_time': total_expected_time,
                    'waste_time': throughput_target - total_expected_time,
                    'machines_required': math.ceil(total_expected_time)
                })
                station_id += 1

    return {'stations': stations}


def generate_summary_report(station_allocation):
    total_stations = len(station_allocation['stations'])
    total_waste_time = sum(station['waste_time'] for station in station_allocation['stations'])
    total_machines_required = sum(station['machines_required'] for station in station_allocation['stations'])

    report = f"""
    Summary Report:
    =================
    - Total Stations: {total_stations}
    - Total Waste Time: {total_waste_time:.2f} minutes
    - Total Machines Required: {total_machines_required}

    Station Details:
    ================
    """
    for station in station_allocation['stations']:
        report += f"""
        Station ID: {station['station_id']}
        - Processes: {[p['id'] for p in station['processes']]}
        - Total Expected Time: {station['total_expected_time']:.2f} minutes
        - Waste Time: {station['waste_time']:.2f} minutes
        - Machines Required: {station['machines_required']}
        """

    return report


def visualize_station_allocation(station_allocation, df):
    """
    Visualize station allocation using a Gantt chart.

    Args:
        station_allocation (dict): Station allocation details.
        df (pd.DataFrame): DataFrame containing process data.

    Returns:
        fig: Matplotlib figure object.
    """
    # Create a mapping of process IDs to their expected time and machine type
    process_info_map = df.set_index('id')[['expectedTimeInMin', 'machineType']].to_dict(orient='index')

    # Get unique machine types and assign a color to each
    machine_types = df['machineType'].unique()
    colors = plt.cm.tab20.colors  # Use a colormap for distinct colors
    machine_color_map = {machine: colors[i % len(colors)] for i, machine in enumerate(machine_types)}

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, station in enumerate(station_allocation['stations']):
        processes = station['processes']
        start_time = 0
        for process in processes:
            process_id = process['id']
            process_info = process_info_map.get(process_id, {'expectedTimeInMin': 0, 'machineType': 'Unknown'})
            expected_time = process_info['expectedTimeInMin']
            machine_type = process_info['machineType']

            # Get the color for the machine type
            color = machine_color_map.get(machine_type, 'gray')

            # Create a horizontal bar for the process
            ax.barh(i, expected_time, left=start_time, color=color, edgecolor='black', label=machine_type)
            
            # Add process name and duration as text inside the bar
            ax.text(start_time + expected_time / 2, i, f"{process['name']}\n({expected_time:.2f} min)", 
                    ha='center', va='center', color='white', fontsize=9, fontweight='bold')

            start_time += expected_time

    # Customize the plot
    ax.set_yticks(range(len(station_allocation['stations'])))
    ax.set_yticklabels([f"Station {station['station_id']}" for station in station_allocation['stations']])
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_title('Station Allocation and Process Timing (Grouped by Machine Type)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Create a legend for machine types
    legend_elements = [patches.Patch(color=color, label=machine) for machine, color in machine_color_map.items()]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title="Machine Type")

    plt.tight_layout()
    return fig


def main(json_file_path, respect_dependencies, max_waste_tolerance=0.1):
    # Load data into a Pandas DataFrame
    df = load_data(json_file_path)
    if df is None:
        return

    # Allocate processes to stations
    station_allocation = allocate_processes_to_stations(df, respect_dependencies=respect_dependencies, throughput_target=1.0)

    # Save station allocation to a JSON file
    station_allocation_file = "station_allocation_output.json"
    save_to_file(station_allocation, station_allocation_file)
    print(f"Station allocation saved to {station_allocation_file}")

    # Generate and save the summary report to a text file
    summary_report = generate_summary_report(station_allocation)
    summary_report_file = "summary_report.txt"
    save_to_file(summary_report, summary_report_file)
    print(f"Summary report saved to {summary_report_file}")

    # Visualize the station allocation and save the plot to a PNG file
    fig = visualize_station_allocation(station_allocation, df)
    visualization_file = "v1.png"
    save_visualization(fig, visualization_file)
    print(f"Visualization saved to {visualization_file}")


if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Optimize jacket production line station allocation.")
    parser.add_argument('json_file', type=str, help="Path to the JSON file containing the process list.")
    parser.add_argument('respect_dependencies', type=bool, help="Whether to respect process dependencies (True/False).")
    parser.add_argument('--max_waste_tolerance', type=float, default=0.1, help="Maximum waste time tolerance in minutes (default: 0.1).")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.json_file, args.respect_dependencies, args.max_waste_tolerance)
