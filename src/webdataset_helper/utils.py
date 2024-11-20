import os
import gzip
import pickle
import yaml
import fsspec

from tqdm.auto import tqdm

from .shard_writer import ShardWriter

def save_hf_data_dataset(
    dataset_name,
    dataset,
    path,
    maxcount=100000,
):
    oinfo = {
        "shard_counts": {}
    }

    osplit = {
        "exclude": [],
        "split_parts": {
            "test": [],
            "val": [],
            "train": [],
        }
    }

    odataset = {
        "__class__": "CrudeWebdataset",
        "__module__": "megatron.energon",
        "subflavors": {
            "data": "data",
        }
    }


    if "train" in dataset:
        with ShardWriter(
            os.path.join(path, 'train-chunk-%d.tar'), 
            maxcount=maxcount,
        ) as shard_writer:
            
            for id, data_item in enumerate(tqdm(dataset['train'], desc='Bar train')):

                sample = {
                    "__key__": f"{dataset_name}-train-{id}",
                    "data": gzip.compress(pickle.dumps(data_item))
                }

                shard_writer.write(sample)

        oinfo['shard_counts'].update({**shard_writer.shard_counts})
        osplit['split_parts']["train"] = sorted({**shard_writer.shard_counts}.keys())

    if "validation" in dataset:
        with ShardWriter(
            os.path.join(path, 'validation-chunk-%d.tar'), 
            maxcount=maxcount,
        ) as shard_writer:
            
            for id, data_item in enumerate(tqdm(dataset['validation'], desc='Bar validation')):

                sample = {
                    "__key__": f"{dataset_name}-validation-{id}",
                    "data": gzip.compress(pickle.dumps(data_item))
                }

                shard_writer.write(sample)

        oinfo['shard_counts'].update({**shard_writer.shard_counts})
        osplit['split_parts']["val"] = sorted({**shard_writer.shard_counts}.keys())

    if "test" in dataset:
        with ShardWriter(
            os.path.join(path, 'test-chunk-%d.tar'), 
            maxcount=maxcount,
        ) as shard_writer:
            
            for id, data_item in enumerate(tqdm(dataset['test'], desc='Bar test')):

                sample = {
                    "__key__": f"{dataset_name}-test-{id}",
                    "data": gzip.compress(pickle.dumps(data_item))
                }

                shard_writer.write(sample)

        oinfo['shard_counts'].update({**shard_writer.shard_counts})
        osplit['split_parts']["test"] = sorted({**shard_writer.shard_counts}.keys())

    with fsspec.open(os.path.join(path, ".nv-meta", ".info.yaml"), 'w', auto_mkdir=True) as yaml_file:
        yaml.dump(oinfo, yaml_file)

    with fsspec.open(os.path.join(path, ".nv-meta", "dataset.yaml"), 'w') as yaml_file:
        yaml.dump(odataset, yaml_file)

    with fsspec.open(os.path.join(path, ".nv-meta", "split.yaml"), 'w') as yaml_file:
        yaml.dump(osplit, yaml_file)





    