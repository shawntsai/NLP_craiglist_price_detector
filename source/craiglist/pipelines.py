# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import nltk
from nltk.tokenize import word_tokenize
import csv

class CraiglistPipeline(object):
    def __init__(self):
        self.id = 0
        
    def process_item(self, item, spider):
        self.id = self.id + 1
        original_data = '\n'.join(item['text'])
        words = word_tokenize(original_data)
        data = nltk.pos_tag(words)
        data = [(A, B, 'O') for (A, B) in data]

        out = open('out' + str(self.id) + '.csv', 'wb')
        csv_out = csv.writer(out)
        for row in data:
            csv_out.writerow(row)

        with open('out' + str(self.id) + '.html', 'wb') as f:
            f.write(item['body'])

