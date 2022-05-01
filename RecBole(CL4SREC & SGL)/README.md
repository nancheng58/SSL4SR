main configuration is in `seq.yaml`

change dataset and model in `run.sh`, (`--model=SGL` or `--model=CL4SRec`)

config file of CL4SRec model is in `recbole/properties/model/CL4SRec.yaml`

the output log file of recbole will be save in log and *.err file if using slurm

converter.ipynb is a jupyter notebook that converts dataset into proper format for recbole (the *.inter file)

run `sbatch run.sh`