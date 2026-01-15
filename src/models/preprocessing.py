# Libs
import pandas as pd

# Modelos
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE

# Classes

# Classe dropa atributos
class DropFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,feature_to_drop = ['ID_Cliente']):
        self.feature_to_drop = feature_to_drop
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feature_to_drop).issubset(df.columns)):
            df.drop(self.feature_to_drop,axis=1,inplace=True)
            return df
        else:
            print('Uma ou mais features não estão no DataFrame')
            return df
        
# Normalização de dados
class MinMax(BaseEstimator,TransformerMixin):
    def __init__(self,min_max_scaler  = ['Idade', 'Rendimento_anual', 'Tamanho_familia', 'Anos_empregado']):
        self.min_max_scaler = min_max_scaler
        self.scaler_fitted = None
    
    def fit(self,df):
        # FIT APENAS em treino (salva limites)
        self.scaler_fitted = MinMaxScaler()
        self.scaler_fitted.fit(df[self.min_max_scaler])
        return self
    
    def transform(self,df):
        # APPLY SEM REFIT (usa limites aprendidos)
        if self.scaler_fitted is None:
            print("Erro: scaler não foi FITado. Execute fit() antes de transform()")
            return df
        
        if (set(self.min_max_scaler).issubset(df.columns)):
            df = df.copy()
            df[self.min_max_scaler] = self.scaler_fitted.transform(df[self.min_max_scaler])
            return df
        else:
            print('Uma ou mais features não estão no DataFrame')
            return df

# Transfmorações de dados OneHotEncoding
class OneHotEncodingNames(BaseEstimator,TransformerMixin):
    def __init__(self,OneHotEncoding = ['Estado_civil', 'Moradia', 'Categoria_de_renda', 
                                        'Ocupacao']):                                      
                                                                               
        self.OneHotEncoding = OneHotEncoding

    def fit(self,df):
        return self

    def transform(self,df):
        if (set(self.OneHotEncoding).issubset(df.columns)):
            # função para one-hot-encoding das features
            def one_hot_enc(df,OneHotEncoding):
                one_hot_enc = OneHotEncoder()
                one_hot_enc.fit(df[OneHotEncoding])
                # obtendo o resultado dos nomes das colunas
                feature_names = one_hot_enc.get_feature_names_out(OneHotEncoding)
                # mudando o array do one hot encoding para um dataframe com os nomes das colunas
                df = pd.DataFrame(one_hot_enc.transform(df[self.OneHotEncoding]).toarray(),
                                  columns= feature_names,index=df.index)
                return df

            # função para concatenar as features com aquelas que não passaram pelo one-hot-encoding
            def concat_with_rest(df,one_hot_enc_df,OneHotEncoding):              
                # get the rest of the features
                outras_features = [feature for feature in df.columns if feature not in OneHotEncoding]
                # concaternar o restante das features com as features que passaram pelo one-hot-encoding
                df_concat = pd.concat([one_hot_enc_df, df[outras_features]],axis=1)
                return df_concat

            # one hot encoded dataframe
            df_OneHotEncoding = one_hot_enc(df,self.OneHotEncoding)

            # retorna o dataframe concatenado
            df_full = concat_with_rest(df, df_OneHotEncoding,self.OneHotEncoding)
            return df_full

        else:
            print('Uma ou mais features não estão no DataFrame')
            return df

# Transformação de OrdinalEncoder
class OrdinalFeature(BaseEstimator,TransformerMixin):
    def __init__(self,ordinal_feature = ['Grau_escolaridade']):
        self.ordinal_feature = ordinal_feature
        # Ordem educacional explícita (não alfabética)
        self.educacao_ordem = {
            'Primário': 1,
            'Secundário': 2,
            'Superior': 3,
            'Pós-Graduação': 4,
            'Nao_informada': 0
        }
    def fit(self,df):
        return self
    def transform(self,df):
        if 'Grau_escolaridade' in df.columns:
            # Usar mapeamento explícito em vez de OrdinalEncoder alfabético
            df['Grau_escolaridade'] = df['Grau_escolaridade'].map(self.educacao_ordem)
            # Investigar valores não mapeados
            if df['Grau_escolaridade'].isna().sum() > 0:
                df['Grau_escolaridade'] = df['Grau_escolaridade'].fillna(0)
            return df
        else:
            print("Grau_escolaridade não está no DataFrame")
            return df

# Classe de balanceamento de dados
class Oversample(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,df):
        return self
    def transform(self,df):
        if 'Mau' in df.columns:
            # função smote para superamostrar a classe minoritária para corrigir os dados desbalanceados
            oversample = SMOTE(sampling_strategy='minority')
            X_bal, y_bal = oversample.fit_resample(df.loc[:, df.columns != 'Mau'], df['Mau'])
            df_bal = pd.concat([pd.DataFrame(X_bal),pd.DataFrame(y_bal)],axis=1)
            return df_bal
        else:
            print("O target não está no DataFrame")
            return df

