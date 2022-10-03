class KNN_Classifier:
    """
    Class function that performs K-Nearest Neighbor classification. 
    It takes two initiation arguments, number of nearest neighbors 
    and distance metric type. It also takes the data set and the 
    class label provided in the fit method as input data to run 
    the class. For instance, the standard way of implementing the 
    function should include the class followed by fit method and 
    predicting method. See example below:
    
    Example:
    --------
    >>> model = KNN_Classifier(k=5, metric_type='manhattan') 
    >>> model.fit(data_set, labels)
    >>> predicted = model.predicting(test_set)

    
    Parameters:
    -----------
    k : number of nearest neighbors to use for classification. 
    This argument can have a range of any 
    integer number not greater than the data size.
    
    metric_type : Is the way of how the distance between data 
    points are measured. The metric type could be either 
    'euclidean' or 'manhattan'.
    
    Methods:
    --------
    ___init__(k=3, metric_type='euclidean')
            Create class initiation with nearest neighbour argument 
            equals 3 and distance metric equals a 'euclidean'.
            
    fit(X, y)
            The method fits the data to run the KNN classifier. 
            See example above.
            
    euclidean(x, x2)
            It measures the distance between two data points using 
            the euclidean method. For more info:
            <https://es.wikipedia.org/wiki/Distancia_euclidiana>_
            
    manhattan(x, x2)
            It measures the distance between two data points using 
            the Manhattan method. It is calculated as the sum of the 
            absolute differences between the two data points.
            
    k_neighbours(self, x)
            Runs the distance methods to calculate the most common 
            neighbours and return the most common class label for given k.
            
    predicting(X)
            It takes a test dataset as an argument and run the 
            k_neighbours method and returns array of predicted class labels.
            
    Error raised:
    -------------
    If k is not an integer it will raise a TypeError:
    If metric type is neither 'euclidean' nor 'manhattan' it will 
    raise a UnboundLocalError:
    If data and class labels are not the same size it will raise a AssertionError:
    If one of the arguments are not provide it raises a TypeError:
    
    """
    # Create class initiation.
    def __init__(self, k, metric_type): 
        self.k = k
        self.metric_type = metric_type
        
    # Defining a method to fit the data to KNN classifier.
    def fit(self, X, y): 
        assert len(X) == len(y) 
        self.X_train = X
        self.y_train = y
    
    # Defining a method that measure euclidean distance between given data points.
    def euclidean(self, x, x2):
        self.x, self.x2 = x, x2 
        squared_length = np.sum((np.array(self.x) - np.array(self.x2))**2)
        return np.sqrt(squared_length)
    
    # Defining a method that measure manhattan distance between given data points.
    def manhattan(self, x, x2):
        self.x, self.x2 = x, x2
        abs_length = np.abs(np.array(self.x) - np.array(self.x2))
        return abs_length.sum()
    
    # Create a method that runs distance functions
    def k_neighbours(self, x):
        if self.metric_type == 'euclidean':
            lengths = [self.euclidean(x, x2) for x2 in self.X_train] 
        elif self.metric_type == 'manhattan':
            lengths = [self.manhattan(x, x2) for x2 in self.X_train]
        k_index = np.argsort(lengths)[: self.k] 
        k_nearest_classes = [self.y_train[i] for i in k_index] 
        count = np.bincount(k_nearest_classes) 
        most_common_class = np.argmax(count) 
        return most_common_class
    
    # Create a method that returns array of predicted class labels.    
    def predicting(self, X): 
        predicted_labels = [self.k_neighbours(x) for x in X]
        return np.array(predicted_labels)