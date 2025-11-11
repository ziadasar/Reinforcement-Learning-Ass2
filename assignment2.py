import wandb

wandb.login()  # enter your API key here
wandb.init(project="my-test-run")
for x in range(10):
    wandb.log({"loss": 10 - x})
