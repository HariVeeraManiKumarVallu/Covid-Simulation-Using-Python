# Covid Simulation Using Python

This repository contains a Python-based simulation for modeling the spread and progression of COVID-19 across different countries and age groups. The simulation uses transition probabilities and holding times to simulate the states of individuals during the pandemic.

## Features
- Simulates COVID-19 progression for different age groups and countries.
- Generates time-series data for infection states.
- Creates visualizations of infection states over time.
- Provides summary statistics for the simulation.

## Files in the Repository
- **`a3-countries.csv`**: Contains population and demographic data for various countries.
- **`a3-covid-simulated-timeseries.csv`**: Simulated time-series data for individual infection states.
- **`a3-covid-summary-timeseries.csv`**: Summary time-series data for infection states across countries.
- **`assignment3.py`**: Main script for running the simulation.
- **`helper.py`**: Helper functions for file handling and plotting.
- **`sim_parameters.py`**: Contains transition probabilities and holding times for different age groups.
- **`test.py`**: Unit tests for the simulation.
- **`README.md`**: Documentation for the project.

## How to Run the Simulation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Covid-Simulation-Using-Python.git
   cd Covid-Simulation-Using-Python
   ```
2. Install the required Python libraries:
   ```bash
   pip install pandas numpy matplotlib
   ```
3. Run the simulation:
   ```bash
   python assignment3.py
   ```
4. To test the simulation:
   ```bash
   python test.py
   ```

## Outputs
- **`a3-covid-simulation.png`**: Visualization of infection states over time for selected countries.
- **Simulated CSV files**: Generated data for individual and summary infection states.

## Dependencies
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`

## License
This project is licensed under the MIT License.

## Author
Hari Veera Mani Kumar Vallu

Feel free to contribute to the project or report any issues!