Code for paper "Successor Uncertainties: Exploration and Uncertainty in Temporal Difference Learning" by David Janz<sup>\*</sup>, Jiri Hron<sup>\*</sup>, Przemysław Mazur, Katja Hofmann, José Miguel Hernández-Lobato, Sebastian Tschiatschek. NeurIPS 2019.
<sup>\*</sup> Equal contribution

Paper is available at https://arxiv.org/abs/1810.06530.

This code allows for reproduction of the tabular experiments. Click [here](https://djanz.org/successor_uncertainties/atari_code) for code to reproduce the Atari 2600 experiments.

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
Figure with results should display to screen, but will also be saved to the figs subfolder.

Instructions to reproduce figure 3 (Chain/Deep Sea experiments) are coming soon.
