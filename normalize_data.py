from sklearn import preprocessing

class NormalizedData():

    def normalize_data(self, data, norm):
    
        if norm is 0:
            x = data[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']]
            n_arr_x = preprocessing.normalize(x, norm='l1')
            return n_arr_x

        elif norm is 1:
            scaler = preprocessing.StandardScaler
            x = data[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']]
            y = data [['OT']]
            scaled_train_data = scaler.fit(x,y)
            n_arr_x = scaler.transform(scaled_train_data)
            return n_arr_x
        
        else:
            print('lol your mom error')      
