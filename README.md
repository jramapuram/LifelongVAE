# Lifelong Variational Autoencoder
A student-teacher variational autoencoder that utilizes a normal
parameterization, coupled with the gumbel reparameterization in order to enforce
consistency across different intervals.

## Vanilla VAE
An example usage of the Vanilla VAE:
```bash
python run_mnist_experiment.py --base_dir="." --device="/gpu:0" --sequential=False --device_percentage=0.9 --latent_size=14 --use_bn=True --epochs=100
```

## Online VAE
An example usage of the Online VAE:
```bash
python run_mnist_experiment.py --base_dir="." --device="/gpu:0" --sequential=True --device_percentage=0.9 --latent_size=14 --use_bn=True --min_interval=12000 --max_dist_swaps=32
```

`--sequential=True` utilizes the online VAE mentioned in the paper as opposed to the vanilla batch method
