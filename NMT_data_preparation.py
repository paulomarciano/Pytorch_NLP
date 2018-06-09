from lxml import etree
from tqdm import tqdm
import pandas as pd

context = etree.iterparse( './OpenSubtitles/en-pt_br.tmx')
en = []
pt = []
mode = 'pt'
s = 1
for event, element in tqdm(context):
    text = element.text
    if text is None:
        if mode =='pt': mode = 'en'
        elif mode =='en': mode = 'pt'
        element.clear()
        s += 1
        flag = True
        continue

    if mode =='pt':
        pt.append(text)
    elif mode =='en':
        if ("\n" in text):
            pass
        else:
            en.append(text)
    element.clear()
    if (s % 1000000 == 0) and flag:
        print(len(en),len(pt))
        pd.DataFrame({'English': en, 'Portuguese': pt}).to_csv('./OpenSubtitles/para_corpus_part_{}.csv'.format(s//1000000),index=False)
        en = []
        pt = []
        flag = False

print(len(en),len(pt))
pd.DataFrame({'English': en, 'Portuguese': pt}).to_csv('./OpenSubtitles/para_corpus_part_{}.csv'.format((s//1000000) + 1),index=False)
en = []
pt = []
flag = False

#pd.DataFrame({'English': en, 'Portuguese': pt}).to_csv('para_corpus.csv')
print('saved')
