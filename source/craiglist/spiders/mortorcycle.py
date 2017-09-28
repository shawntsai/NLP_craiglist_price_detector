# -*- coding: utf-8 -*-
import scrapy


class MortorcycleSpider(scrapy.Spider):
    name = 'mortorcycle'
    start_urls = ['https://losangeles.craigslist.org/search/mca', 
                  'https://losangeles.craigslist.org/search/mca?s=120', 
                  'https://losangeles.craigslist.org/search/mca?s=240', 
                  'https://losangeles.craigslist.org/search/mca?s=360']
    

    def parse(self, response):
        next_pages = response.css('.hdrlnk::attr(href)').extract()
        for next_page in next_pages:
            yield scrapy.Request(next_page.encode('utf-8'), callback=self.parsePage)

	
    def parsePage(self, response):
        content = response.css('#postingbody::text').extract()
        yield {'text': content, 'body': response.body}
        
        

