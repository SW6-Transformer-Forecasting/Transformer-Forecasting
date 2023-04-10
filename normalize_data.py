from sklearn import preprocessing

class NormalizedData():

    def normalize_data(self, data, norm):
    
        if norm == 0:
            x = data[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']]
            n_arr_x = preprocessing.normalize(x, norm='l1')
            return n_arr_x

        elif norm == 1:
            x = data[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']]
            y = data [['OT']]
            scaler = preprocessing.StandardScaler()
            scaler.fit(x)
            n_arr_x = scaler.transform(x)
            return n_arr_x
        
        else:
            print("Normalization error - No normalizer found")
