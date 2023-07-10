import os

STR2GPU = {'00': 'MIG-f3304720-4601-5894-bee4-cd0174024e06',
           '10': 'MIG-35d812cd-2b57-5e55-bb30-890bd9675846',
           '20': 'MIG-dc45e153-fb1e-5b2d-8a31-d3fb9494cd80',
           '21': 'MIG-21d343f4-de6e-5d44-9774-e2f3dbab968d',
           '30': 'MIG-0b2452d4-9b27-530f-a6f1-1c2d05dfaa72',
           '31': 'MIG-e46a8085-268f-5417-8e5a-a9e20578424d'}


def set_gpu(config):
    if config.server == 'workstation2':
        gpus = ','.join([STR2GPU[str(gpu)] for gpu in config.gpus])
    else:
        gpus = ','.join([str(gpu) for gpu in config.gpus])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Data Distributed Training", add_help=False)
    parser.add_argument('--gpus', type=str, nargs='+', default=None, help='')
    parser.add_argument('--server', type=str)

    config = parser.parse_args()
    set_gpu(config)
