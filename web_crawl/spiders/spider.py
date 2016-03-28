import scrapy
from web_crawl.items import WebCrawlItem
from scrapy.selector import HtmlXPathSelector


class NYTimesSpider(scrapy.Spider):
    name = "article"
    start_urls = [
        "http://www.nytimes.com/pages/world/index.html?action=click&pgtype=Homepage&region=TopBar&module=HPMiniNav&contentCollection=World&WT.nav=page"]

    def parse(self, response):
        items = []
        hxs = HtmlXPathSelector(response)
        sites = hxs.select('//div[@class="story"]')
        for sel in sites:
            item = WebCrawlItem()
            item['title'] = sel.select('.//h3/a/text()').extract()
            item['content'] = sel.select(
                './/p[@class="summary"]/text()').extract()
            item['link'] = sel.select('.//h3/a/@href').extract()
            items.append(item)
        return items
