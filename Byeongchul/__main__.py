from model import Algorithm

HParams = {
    'input': ['Orifice diameter', 'Time 1', 'PPM 1', 'Time 2', 'PPM 2', 'Effect', 'Power'],
    'output': ['Effect', 'Power'],
    'dir_data': "data/Consequence_211104_mod.csv",
    'dir_model': "data/orifice_weight/",
    'dir_log': "data/Learning_log/",
    'dir_result': "data/result_test_Servelity.csv",
    'epochs': 300,
    'decay_steps': 3000,
    'decay_rate': 0.9,
    'lr': 0.0001,
    'random_state': 1,
    'batch_size': 1
}

def main():
    a = Algorithm(HParams)
    a.trainRun(is_trans=False)
    a.testRun()

if __name__ == '__main__':
  main()