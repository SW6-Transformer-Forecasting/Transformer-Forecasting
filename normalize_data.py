from sklearn import preprocessing


class NormalizedData():

    def normalize_data(self, data):

        X = data[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']]

        n_arr_x = preprocessing.normalize(X, norm='l1')

        return n_arr_x
