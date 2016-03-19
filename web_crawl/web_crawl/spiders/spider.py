from scrapy.spider import BaseSpider
from web_crawl.items import WebCrawlItem
from scrapy.selector import HtmlXPathSelector


class MySpider(BaseSpider):
    name = "article"
    start_urls = [
        "http://www.nytimes.com/pages/world/index.html?action=click&pgtype=Homepage&region=TopBar&module=HPMiniNav&contentCollection=World&WT.nav=page"]

    def parse(self, response):
        items = []
        hxs = HtmlXPathSelector(response)
        for sel in hxs.select('//div[@class="story"]'):
            item = WebCrawlItem()
            item['content'] = sel.select('//p').extract()
            item['title'] = sel.select('//h3').extract()
            item['link'] = sel.select('//a/@href').extract()
            items.append(item)
        return items
