# Successor Uncertainties, Tabular Experiments

This code allows for reproduction of the tabular experiments in https://arxiv.org/abs/1810.06530. Click [here](https://djanz.org/successor_uncertainties/atari_code) for code to reproduce the Atari 2600 experiments.

Clone && pip install the requirements. 

To reproduce results for the Tree MDP, run 
```
python3 run 5 specs/name_of_spec.json
```
where name_of_spec is one of tree250_su, tree250_boot1x or tree250_boot25x. The json files contain the settings for each run configuration. Successor Uncertainties and Bootstrap with 1x computation should finish in minutes. Bootstrap with 25x compute may take considerable time.

Output will be saved to the data_out folder. To plot results (as in figure 2 in the paper), run
```
python3 plotting/plot_scaling.py --file data_out/name_of_file.pkl --show
```
Figure with results should display to screen, but will also be saved to figs/scaling.pdf.

To reproduce Chain/Deep Sea experiments (figure 3 in paper) run
```
python3 run 5 specs/deepsea_su.json
```
Note, however, this might take a while and requires a lot of RAM. Edit deepsea_su.json and change env_size values to test on smaller versions of the MDP.

Then, to plot the results, run
```
python3 plotting/plot_scaling.py --file data_out/su_deepsea_out.pkl --loglog --show
```
Again, plot will be saved to figs/scaling_loglog.pdf. Orange line in the resulting plot is taken directly from Osband et al. 2018.
