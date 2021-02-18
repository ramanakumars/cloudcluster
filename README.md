# Clustering cloud types from JunoCam images
This is a tool to run the projection and subsequent classification from JunoCam images. 

## Dependencies
Install the python dependencies using the `requirements.txt`:

```bash
pip3 install -r requirements.txt
```

## Installation
To run the projection code, the C extension needs to be compiled. To do this, run,
```bash
cd projection/
make clean
make
```

## Examples
See `projection.ipynb` for an example of JunoCam image projection.

See `do_cluster.ipynb` for an example of cloud clustering from a map projected JunoCam image.

To rerun the projection code, you will need to unzip the `5989-Data.zip` and `5989-ImageSet.zip`
files, which will create the `DataSet` and `ImageSet` directories. 
