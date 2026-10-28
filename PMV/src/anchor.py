from pathlib import Path

def src_dir():
    return str(Path(__file__).parent)


def data_dir():
    data_path = Path(__file__).parent / 'data' #Path(__file__).parent / '..' / 'data'
    data_path.mkdir(parents=True, exist_ok=True)
    return str(data_path)

def experiment_dir():
    experiment_path = Path(__file__).parent / 'Experiments' 
    return(str(experiment_path))