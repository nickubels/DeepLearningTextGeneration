from fastai.text import *


path = untar_data(URLs.IMDB_SAMPLE)
path.ls()

df = pd.read_csv(path/'texts.csv')
df.head()

df['text'][1]

data_lm = TextDataBunch.from_csv(path, 'texts.csv')

data_lm.save()

data = load_data(path)

data = TextClasDataBunch.from_csv(path, 'texts.csv')
data.show_batch()
