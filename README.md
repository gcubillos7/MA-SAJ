# MA-SAJ
For continuous version visit [MA-SAJ-Continuous](https://github.com/gcubillos7/MA-SAJ-Continuous).
This implementation is written in PyTorch and is based on PyMARL and SMAC.

Installation instructions
Build the Dockerfile using

bash docker/build.sh
Set up StarCraft II and SMAC:

bash install_sc2.sh
This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment.

Run an experiment
python3 src/main.py --config=masaj --env-config=sc2 with env_args.map_name=MMM2 n_roles=3 role_interval=5 t_max=2050000
To change the annealing time of epsilon, set epsilon_anneal_time_exp.

All results will be stored in the Results folder.

Saving and loading learnt models
Saving models
You can save the learnt models to disk by setting save_model = True, which is set to False by default. The frequency of saving models can be adjusted using save_model_interval configuration. Models will be saved in the result directory, under the folder called models. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

Loading models
Learnt models can be loaded using the checkpoint_path parameter, after which the learning will proceed from the corresponding timestep.

Watching StarCraft II replays
save_replay option allows saving replays of models which are loaded using checkpoint_path. Once the model is successfully loaded, test_nepisode number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., runner=episode. The name of the saved replay file starts with the given env_args.save_replay_prefix (map_name if empty), followed by the current timestamp.

Note: Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.
