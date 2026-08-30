# Official data provenance

## MAGNET

Idaho National Laboratory `MAGNET-Heat-Pipe-Data` public archive.

Files used:

- `Experiment/Single_File/MAGNET_Heat_Pipe_2022-03-30.csv`
- `Machine_Learning/Single_File/ML_MAGNET_2022-03-30.csv`

The repository's existing MAGNET audit already uses these exact files.

## FreeTwinEV 1S4P

Zenodo record **19935693**, DOI **10.5281/zenodo.19935693**.

File: `FTEV_1S4P_IdentValid_SimPackage_V1.zip`

Official MD5 frozen in the runner: `a37cf1619d14a28e4e4397b55f0b58ac`.

The package contains the ID22 experiment and matching cooldown/discharge 3D CFD validation simulation outputs.

## TU Wien SNG

TU Wien Research Data record **6mmjq-1tj37**, DOI **10.48436/6mmjq-1tj37**.

Files used:

- `Data_MPC_DFB.csv`
- `Data_MPC_Syngas.csv`
- `Data_SoftSensor.csv`
- `README.txt`

Official MD5 values are frozen in the analysis script. `Data_IPSE.json` is deliberately not required for this first contract-transfer test because the DFB estimator and product-gas soft sensor already expose explicit physical/virtual pairs with clear semantic correspondence.
