import os
import re
import shutil

BASE_DIR = '/home/jingyihe/scratch/sample_mle'
TEST_DIR = '/home/jingyihe/scratch/test_emb'
if __name__ == '__main__':
    # # if os.path.exists(TEST_DIR):
    # #     os.rmdir(TEST_DIR)
    # if not os.path.exists(TEST_DIR):
    #     os.makedirs(os.path.join(TEST_DIR,'set_of_emb'), exist_ok=True)
    # for i in range(100):
    #     os.makedirs(os.path.join(TEST_DIR, 'set_of_emb', str(i*100), 'dictionary'),exist_ok=True)
    #     with open(os.path.join(TEST_DIR, 'set_of_emb', str(i*100), 'a.txt'), 'w') as text:
    #         text.writelines('test\ntest')
    #
    # with open(os.path.join(TEST_DIR,'set_of_emb','trace.txt'), 'w') as f:
    #     f.writelines('hahahahaha')

    embedding_lst = os.listdir(BASE_DIR)
    for emb in embedding_lst:
        listFile = [f for f in os.listdir(os.path.join(BASE_DIR, emb)) if re.match(r'^\d+', f)]
        try:
            save_file = str(max(map(int, listFile)))
        except ValueError:
            print("no embedding was saved for experiment {}, bad experiment.".format(emb))
            continue
        print("{} epoch is saved for {} experiment".format(save_file,emb))
        for f in listFile:
            if f != save_file:
                shutil.rmtree(os.path.join(BASE_DIR,emb,f))
