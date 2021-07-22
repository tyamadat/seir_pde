# SEIR PDE

SEIR PDE estimates effective distance based on time series data of infection, enabling us to understand infection-related human mobility networks. 

## Manuscript (preprint)
Tetsuya Yamada* and Shoi Shi*, Estimating infection-related human mobility networks based on time series data of COVID-19 infection in Japan. bioRxiv. 

## Requirements
This project requires the following libraries.
- NumPy
- SciPy
- Pandas
- odeintw > 0.1.0
- emcee > 3.0.0
- corner > 2.2.0
- tqdm

## Folder structure
### `src`
All source codes used in our manuscript are in this folder. 
- `models.py`: 
    - The SEIR model expressed by ordinary differential equations (ODE)
    - The diffusion model expressed by partial differential equations (PDE). 
    - The diffusion model that was used for estimating impacts of the effective distance on the scale of the pandemic. 
    - The diffusion model in inter-prefecture network graph. 
- `mcmc.py`:\
Run Markov chain Monte Carlo (MCMC) using affine invariant methods to estimate parameters, judge convergence based on an auto-correlation function, and visualize a result using estimated parameters. 
- `cartogram.py`\
Distort a map based on the effective distance from a reference point (e.g., Tokyo) and local connectivity such as geographical distance between nearby prefectures. 

### `cli`
You can use command line interface to run source codes in `src`. 

### `data`
All data used in our manuscript are in this folder. 
- `distance.xlsx`:\
Geographical distance between any two prefectures. 
- `gadm36_JPN.gpkg`:\
GeoPackage data of the entire Japan, downloaded from Database of Global Administrative Areas (GADM). 
- `passenger_traffic_2019.xlsx`:\
The survey data of passenger traffic between prefectures in 2019, provided by Ministry of Land, Infrastructure, Transport, and Tourism. 
- `population.csv`:\
Population in each prefecture, provided by Statistics Bureau of Japan. 
- `prefectures.csv`:\
Infection status by prefecture, provided by [Toyo Keizai Inc](https://toyokeizai.net/sp/visual/tko/covid19/en.html). 

## 