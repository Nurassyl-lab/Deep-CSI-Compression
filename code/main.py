'''
    main file for the project

    here you can choose the type of the test you want to run

'''

import argparse

# using argparse get the args
parser = argparse.ArgumentParser(description='Run the test')
parser.add_argument('--test', type=str, default='test1', help='choose the test you want to run')
args = parser.parse_args()



if __name__ == '__main__':
    print(f'Running the test --> {args.test}')
