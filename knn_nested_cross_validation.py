class KNN_NestedCrossV(KNN_Classifier):
    """
    Class function that performs nested cross validation with KNN classification. It uses the KNN_Classifer as 
    inherited class function to compute best grid and train models.
    It takes five initiation arguments, number of nearest neighbors, distance metric type, index fold splitting, 
    size of the splitting data and random seed for data permutation. It also takes the data set and the class label 
    provided in the fit method as input data to run the class. For
    instance, the standard way of implementing the class function should include the class followed by fit method and 
    displaying results method. See example below:
    
    Example:
    --------
    >>> model = KNN_NestedCrossV(k, metric_type, nfolds, split_size, seed)
    >>> model.fit(data_set, labels)
    >>> model.disply_results()

    
    Parameters:
    -----------
    k : number of nearest neighbors to use for classification. This argument can have a range of any 
    integer number not greater than the data size.
    
    metric_type : Is the way of how the distance between data points are measured. 
    The metric type could be either 'euclidean' or 'manhattan'.
    
    nfolds : Defines the value for the split of the data indices. It could be only an integer otherwise it will raise an error.
    See Error raised section below.
    
    seed : seed could be any integer number that will represent the combination of random permutation. For more 
    details see numpy.random.seed on: <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.seed.html>
    
    best_params : A list for dictionaries with all argioments for the grid search.
    
    i : integer to enumerate loops.
    
    Methods:
    --------
    ___init__(k, metric_type, nfolds, split_size, seed)
            Create class initiation with all the parameters described above.
    fit(X, y)
            The method fits the data to run the KNN nested cross validation class. It takes two arguments, 
            data and labels.See example above.
    nested_crossV()
            It loops indices through all the permutations necessary to cross validate the datasets.
    inner_loop(self, X_train, y_train, X_valid, y_valid, X_test, y_test)
            The method takes all the dataset from the nested cross validation and loops through to defines best 
            knn grid parameters.
    outer_loop(self, X_valid, y_valid, X_test, y_test)
            It loops all the validation datasets to test them and determine the best validation set per round.
    disply_results()
            It is the outer loop that cycles through all sequences to display results necessary.
            
    Error raised:
    -------------
    If k is not an integer it will raise a TypeError:
    If metric type is neither 'euclidean' nor 'manhattan' it will raise a UnboundLocalError:
    If nfolds is not an integer it will raise a TypeError:
    If data size is not an integer it will raise a TypeError:
    If data and class labels are not the same size it will raise a AssertionError:
    If one of the arguments are not provide it raises a TypeError:
    
    """
    
    # create class initiation with K, distance function, cross validation and data size splitting.
    def __init__(self, k, metric_type, nfolds, seed):
        self.k = k
        self.metric_type = metric_type
        self.nfolds = nfolds
        self.seed = seed
        self.best_params = []
        self.KNN_Classifier = KNN_Classifier
        self.i = int
        
    # defining a function to fit the data to the class.
    def fit(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        
    # creating a function to perform cross validation. 
    def nested_crossV(self):
        #splitting data into random permutation of equal parts, redundant data points are excluded.
        np.random.seed(self.seed)
        equal_parts = np.array(len(self.X)/self.nfolds, dtype=int)
        index = np.random.permutation(np.arange(0,len(self.X),1))
        index = np.random.choice(index, self.nfolds*equal_parts)
        index = np.array_split(index, self.nfolds)
        
        # determining cross validation split and combinatins for inner and outer, training test sets. 
        train_fold_inner, valid_fold, train_fold_outer, test_fold  = [],[],[],[]
        for n in range(0, self.nfolds):
            test_fold.append(index[n])
            remaining_folds = np.delete(range(0, self.nfolds), n)
            train_fold_temp1 = []
            for r in remaining_folds:
                train_fold_temp1.extend(index[r])
                valid_fold.append(index[r])
                left_folds = np.delete(range(0, self.nfolds), [n, r])
                train_fold_temp2 = []
                for l in left_folds:
                    train_fold_temp2.extend(index[l])
                train_fold_inner.append(train_fold_temp2)
            train_fold_outer.append(train_fold_temp1)
                
        # converting index into datasets for the inner loop.
        X_train_in = np.array([self.X[i] for i in train_fold_inner])
        y_train_in = np.array([self.y[i] for i in train_fold_inner])
        X_valid_in = np.array([self.X[i] for i in valid_fold])
        y_valid_in = np.array([self.y[i] for i in valid_fold])
        
        # converting index into datasets for the outer loop.
        X_train_out = np.array([self.X[i] for i in train_fold_outer])
        y_train_out = np.array([self.y[i] for i in train_fold_outer])
        X_test_out  = np.array([self.X[i] for i in test_fold])
        y_test_out  = np.array([self.y[i] for i in test_fold]) 
        return  zip(X_train_in, y_train_in, X_valid_in, y_valid_in), zip(X_train_out, y_train_out, X_test_out, y_test_out)
    
    # creating a method for the inner loop that cycles through all the KNN parameters for the best grid.                    
    def inner_loop(self, X_train, y_train, X_test, y_test):
        best_acc = 0
        k_range = self.k
        mt_range = self.metric_type
        euclidean_fold, manhattan_fold = [],[]
        best_grid = {'k': None, 'metric_type':None, 'accuracy_mean':None}

        # looping KNN arguments for grid search.
        for kv in k_range:
            for mt in mt_range:
                classifier = self.KNN_Classifier(k=kv, metric_type=mt)
                classifier.fit(X_train, y_train) 
                predicted = classifier.predicting(X_test) 
                accuracy = (np.sum(predicted == y_test))/len(y_test)
                
                # computing average mean score for each dataset
                if mt == 'euclidean':
                    euclidean_fold.append(accuracy)
                    best_grid['accuracy_mean'] = np.mean(euclidean_fold)
                else:
                    manhattan_fold.append(accuracy)
                    best_grid['accuracy_mean'] = np.mean(manhattan_fold)
                    
                # rate accuracy of the classifier for each parameter.    
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_grid['k'] = kv
                    best_grid['metric_type'] = mt
        self.best_params.append(best_grid)
                           
        # printing out results for the validation rounds.
        table = pd.DataFrame({'Euclidiean_accuracy': euclidean_fold,
                            'Manhattan_accuracy': manhattan_fold,                        
                            'Best_k':best_grid['k'],
                            'Best_metric_type': best_grid['metric_type'],
                            'Best_accuracy': best_acc,
                            'Average_accuracy': best_grid['accuracy_mean']}).round(3)
        table.index.name='k'
        table.index+=1 

        # adjusting jupyter notebook to display dataframes, one after another.
        from IPython.display import display, HTML
        css = """.output {flex-direction: row;}"""
        HTML('<style>{}</style>'.format(css))
        return display(table)
    
    # Defining a method for train test with the best parameters from the inner loop. 
    def outer_loop(self, X_train, y_train, X_test, y_test):
        
        # determining the best parameter grid.
        best_grid = {'k': None, 'metric_type':None, 'accuracy':None, 'Model':None}
        best_grid1 = sorted(self.best_params, key=lambda d: -d['accuracy_mean'])

        # runnig best model arguments to train the outer loop.
        classifier = self.KNN_Classifier(k=best_grid1[0]['k'], metric_type=best_grid1[0]['metric_type']) 
        classifier.fit(X_train, y_train)
        predicted = classifier.predicting(X_test)
        accuracy = (np.sum(predicted == y_test))/len(y_test) 
        best_grid['accuracy'] = np.round_(accuracy, 3)
        best_grid['k'] = best_grid1[0]['k']
        best_grid['metric_type'] = best_grid1[0]['metric_type']

        # printing out the model type according to data split.
        if self.i == 0: 
            best_grid['Model']=('10000')
        elif self.i==1: 
            best_grid['Model']=('01000')
        elif self.i==2: 
            best_grid['Model']=('00100')
        elif self.i==3: 
            best_grid['Model']=('00010')
        elif self.i==4: 
            best_grid['Model']=('00001') 

        # displaying confusion matrix for the best predictions.
        y_predict = pd.Series(predicted)
        y_actual = pd.Series(y_test)
        cm = pd.crosstab(y_actual, y_predict, 
                         rownames=['Actual'], 
                         colnames=['Predicted'], margins=True)
        print('\n', 'Parameters : ''{}'.format(best_grid))
        
        # computing common measures from the confusion matrices: precison, recall, f-score, (floating-point errors are handled also). 
        np.seterr(divide='ignore', invalid='ignore')
        cm_ = cm.values
        TP = cm_.diagonal()[:-1]
        accuracy  =  TP / cm_[ -1, -1]
        precision =  TP / cm_[  3,:-1]
        recall    =  TP / cm_[:-1,  3]
        f_score   =  (2 * precision * recall) / (precision + recall)
        overall_accuracy = sum(accuracy)

        # creating pandas format dataframe to disply results.
        measures = pd.DataFrame({'Accuracy': accuracy, 
                            'Precision': precision, 
                            'Recall': recall,
                            'F-score': f_score,                    
                            'Predicted':cm_[:-1,  3],
                            'Actual':cm_[  3,:-1],
                            'Total': cm_[ -1, -1],
                            'Model_Accuracy':overall_accuracy}).round(3)                        
        measures.index.name='Class_Label'
        return display(cm), display(measures)
                    
    # Creating a method to run the inner and outer loop and display the required results.
    def disply_results(self):
        print('\n VALIDATION RESULTS(INNER LOOP):')
        
        # running the inner loop to get data tables for the best validation set of each round.        
        for self.i, (X_train, y_train, X_test, y_test) in enumerate(self.nested_crossV()[0]):
            tables = self.inner_loop(X_train, y_train, X_test, y_test)
            
        # displaying top 5 validation results.
        best_folds = sorted(self.best_params, key=lambda d: -d['accuracy_mean'])
        print('\n TOP 5 VALIDATION PARAMETERS:\n',
                pd.DataFrame(best_folds[0:5]).to_string(index=False), 
                 '\n\n\n', 'CONFUSION MATRICES(OUTER LOOP):\n')
        
        # running outer loop for the best data model with the best grid parameters.
        for self.i, (X_train, y_train, X_test, y_test) in enumerate(self.nested_crossV()[1]):
            cm , measures = self.outer_loop(X_train, y_train, X_test, y_test)
            
        return tables, cm, measures