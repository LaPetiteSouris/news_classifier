# Scrapy settings for web_crawl project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/topics/settings.html
#

BOT_NAME = 'web_crawl'
BOT_VERSION = '1.0'

SPIDER_MODULES = ['web_crawl.spiders']
NEWSPIDER_MODULE = 'web_crawl.spiders'
USER_AGENT = '%s/%s' % (BOT_NAME, BOT_VERSION)

FEED_FORMAT = 'jsonlines'
FEED_URI = 'web_crawl/result.jl'
