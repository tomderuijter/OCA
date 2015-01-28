'''
Created on May 8, 2012

@author: Tom de Ruijter
'''


def get_set(dataset):
    if dataset == 'a8a':
        args = ['data/a8a/a8a_train.txt',
                'data/a8a/a8a_test.txt',
                'data/a8a/a8a_unlabeled.txt']
        N = 6512
        D = 123
    
    #TODO: Remove me.
    if dataset == 'a8a_test':
        args = ['data/a8a/test/a8a_train.txt',
                'data/a8a/test/a8a_test.txt',
                'data/a8a/test/a8a_unlabeled.txt']
        N = 3256
        D = 123

    elif dataset == 'abalone':
        args = ['data/abalone/abalone_train.txt',
                'data/abalone/abalone_test.txt',
                'data/abalone/abalone_unlabeled.txt']
        N = 835
        D = 8

    elif dataset == 'australian':
        args = ['data/australian/australian_train.txt',
                'data/australian/australian_test.txt',
                'data/australian/australian_unlabeled.txt']
        N = 118
        D = 14
    
    #TODO: Remove set below.
    elif dataset == 'australian_test':
        args = ['data/australian/test/australian_train.txt',
                'data/australian/test/australian_test.txt',
                'data/australian/test/australian_unlabeled.txt']
        N = 69
        D = 14
    
    elif dataset == 'breastcancer':
        args = ['data/breastcancer/breastcancer_train.txt',
                'data/breastcancer/breastcancer_test.txt',
                'data/breastcancer/breastcancer_unlabeled.txt']
        N = 136
        D = 10

    elif dataset == 'cadata':
        args = ['data/cadata/cadata_train.txt',
                'data/cadata/cadata_test.txt',
                'data/cadata/cadata_unlabeled.txt']
        N = 4128
        D = 8

    elif dataset == 'codrna':
        args = ['data/codrna/codrna_train.txt',
                'data/codrna/codrna_test.txt',
                'data/codrna/codrna_unlabeled.txt']
        N = 11907
        D = 8

    elif dataset == 'covtype':
        args = ['data/covtype/covtype_train.txt',
                'data/covtype/covtype_test.txt',
                'data/covtype/covtype_unlabeled.txt']
        N = 12000
        D = 54
    
    elif dataset == 'cpusmall':
        args = ['data/cpusmall/cpusmall_train.txt', 
                'data/cpusmall/cpusmall_test.txt', 
                'data/cpusmall/cpusmall_unlabeled.txt']
        N = 1638
        D = 12

    elif dataset == 'diabetes':
        args = ['data/diabetes/diabetes_train.txt',
                'data/diabetes/diabetes_test.txt',
                'data/diabetes/diabetes_unlabeled.txt']
        N = 153
        D = 8
        
    elif dataset == 'epsilon':
        args = ['data/epsilon/epsilon_train.txt',
                'data/epsilon/epsilon_test.txt',
                'data/epsilon/epsilon_unlabeled.txt']
        N = 2000
        D = 2000

    elif dataset == 'germannumer':
        args = ['data/germannumer/germannumer_train.txt',
                'data/germannumer/germannumer_test.txt',
                'data/germannumer/germannumer_unlabeled.txt']
        N = 200
        D = 24

    elif dataset == 'gisette':
        args = ['data/gisette/gisette_train.txt',
                'data/gisette/gisette_test.txt',
                'data/gisette/gisette_unlabeled.txt']
        N = 1400
        D = 5000

    elif dataset == 'housing':
        args = ['data/housing/housing_train.txt',
                'data/housing/housing_test.txt',
                'data/housing/housing_unlabeled.txt']
        N = 101
        D = 13
    
    elif dataset == 'ijcnn1':
        args = ['data/ijcnn1/ijcnn1_train.txt',
                'data/ijcnn1/ijcnn1_test.txt',
                'data/ijcnn1/ijcnn1_unlabeled.txt']
        N = 9998
        D = 22
        
    elif dataset == 'kdda':
        args = ['data/kdda/kdda_train.txt',
                'data/kdda/kdda_test.txt',
                'data/kdda/kdda_unlabeled.txt']
        N = 12000
        D = 20216830
    
    elif dataset == 'mg':
        args = ['data/mg/mg_train.txt',
                'data/mg/mg_test.txt',
                'data/mg/mg_unlabeled.txt']
        N = 277
        D = 6
    
    elif dataset.split('_')[0] == 'mnist':
        path = 'data/mnist/' + dataset.split('_')[1] + '/'
        args = [path + 'mnist_' + dataset.split('_')[1] + '_train.txt',
                path + 'mnist_' + dataset.split('_')[1] + '_test.txt',
                path + 'mnist_' + dataset.split('_')[1] + '_unlabeled.txt']
        N = 12000
        D = 780
        
    elif dataset == 'mushrooms':
        args = ['data/mushrooms/mushrooms_train.txt',
                'data/mushrooms/mushrooms_test.txt',
                'data/mushrooms/mushrooms_unlabeled.txt']
        N = 1624
        D = 112

    elif dataset == 'news':
        args = ['data/news/news_train.txt',
                'data/news/news_test.txt',
                'data/news/news_unlabeled.txt']
        N = 3999
        D = 1355191
        
    elif dataset.split('_')[0] == 'newsmc':
        path = 'data/newsmc/' + dataset.split('_')[1] + '/'
        args = [path + 'news20scale_' + dataset.split('_')[1] + '_train.txt',
                path + 'news20scale_' + dataset.split('_')[1] + '_test.txt',
                path + 'news20scale_' + dataset.split('_')[1] + '_unlabeled.txt']
        N = 3985
        D = 62061
    
    elif dataset == 'prdata':
        args = ['data/prdata/prdata_train.txt',
                'data/prdata/prdata_test.txt',
                'data/prdata/prdata_unlabeled.txt']
        N = 639
        D = 201740

    elif dataset == 'rcv1':
        args = ['data/rcv1/rcv1_train.txt',
                'data/rcv1/rcv1_test.txt',
                'data/rcv1/rcv1_unlabeled.txt']
        N = 4048
        D = 47236
        
    elif dataset == 'realsim':
        args = ['data/realsim/realsim_train.txt',
                'data/realsim/realsim_test.txt',
                'data/realsim/realsim_unlabeled.txt']
        N = 14461
        D = 20958
        
    elif dataset.split('_')[0] == 'satimage':
        path = 'data/satimage/' + dataset.split('_')[1] + '/'
        args = [path + 'satimage_' + dataset.split('_')[1] + '_train.txt',
                path + 'satimage_' + dataset.split('_')[1] + '_test.txt',
                path + 'satimage_' + dataset.split('_')[1] + '_unlabeled.txt']
        N = 1287
        D = 36
    
    elif dataset == 'space':
        args = ['data/space/space_train.txt',
                'data/space/space_test.txt',
                'data/space/space_unlabeled.txt']
        N = 621
        D = 6
    
    elif dataset == 'svmguide1':
        args = ['data/svmguide1/svmguide1_train.txt',
                'data/svmguide1/svmguide1_test.txt',
                'data/svmguide1/svmguide1_unlabeled.txt']
        N = 1417
        D = 4
        
    elif dataset == 'svmguide3':
        args = ['data/svmguide3/svmguide3_train.txt',
                'data/svmguide3/svmguide3_test.txt',
                'data/svmguide3/svmguide3_unlabeled.txt']
        N = 256
        D = 21

    elif dataset == 'w8a':
        args = ['data/w8a/w8a_train.txt',
                'data/w8a/w8a_test.txt',
                'data/w8a/w8a_unlabeled.txt']
        N = 12940
        D = 300
    
    #TODO: Remove me.
    elif dataset == 'w8a_test':
        args = ['data/w8a/test/w8a_small_train.txt',
                'data/w8a/test/w8a_small_test.txt',
                'data/w8a/test/w8a_small_unlabeled.txt']
        N = 12940
        D = 300

    elif dataset == 'url':
        args = ['data/url/url_train.txt',
                'data/url/url_test.txt',
                'data/url/url_unlabeled.txt']
        N = 12000
        D = 3231961

    elif dataset == 'webspam':
        args = ['data/webspam/webspamuni_train.txt',
                'data/webspam/webspamuni_test.txt',
                'data/webspam/webspamuni_unlabeled.txt']
        N = 12000
        D = 16609143

    elif dataset == 'yearpred':
        args = ['data/yearpred/yearpred_train.txt',
                'data/yearpred/yearpred_test.txt',
                '']
        N = 463715
        D = 90
    
    return args, N, D