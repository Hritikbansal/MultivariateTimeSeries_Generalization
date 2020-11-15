import sys, os

root = '/data/pankaj/CLTS/output/CombGen/'
#root = '/home/hw1415904/data3/codes/CL4MTS/18april_perNodeCL_dropout/CLTS/output/CombGen/'



folder_name=root+sys.argv[1]
test_type=sys.argv[2] # can be iid, ood, or sensitivitytest

collate_results_file = open(root+'/'+sys.argv[1]+'_'+test_type+'_collate_results_file.csv','a')
#collate_results_file.write('\n')

for folder1 in os.listdir(folder_name):
    print(folder1)
    for folder2 in os.listdir(folder_name+'/'+folder1):
        print(folder2)
        for file_name in os.listdir(folder_name+'/'+folder1+'/'+folder2):
            print(file_name)
            if 'numbers' in file_name and '.csv' in file_name and test_type in file_name:
                f = open(folder_name+'/'+folder1+'/'+folder2+'/'+file_name,'r')
                collate_results_file.write(','.join([folder_name,folder1,folder2,file_name,f.readline()]))
                collate_results_file.write('\n')

collate_results_file.close()
