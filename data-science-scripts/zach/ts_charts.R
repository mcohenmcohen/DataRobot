library(httr)
x= POST(
  'https://www.iso-ne.com/isoexpress/web/reports/load-and-demand?p_p_id=operdataviewdetails_WAR_isoneoperdataviewportlet&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view&p_p_cacheability=cacheLevelPage&p_p_col_id=column-2&p_p_col_count=1',
  encod='json',
  body=list(
    '_operdataviewdetails_WAR_isoneoperdataviewportlet_treenode'='dmnd',
    '_operdataviewdetails_WAR_isoneoperdataviewportlet_fileName'='dmnd',
    'p_p_resource_id'='downloadHistZips',
    '_operdataviewdetails_WAR_isoneoperdataviewportlet_reportId'='014',
    '_operdataviewdetails_WAR_isoneoperdataviewportlet_from'='06/20/2011',
    '_operdataviewdetails_WAR_isoneoperdataviewportlet_to'='06/20/2017',
    '_operdataviewdetails_WAR_isoneoperdataviewportlet_captchaText'='2633',
    'p_p_resource_id'='validateCaptcha'
  )
)

N <- 100
x <- runif(N)
y <- runif(N)
cols <- c("red", "black")
color <- sample(cols, N, replace=T, prob=c(.1, .9))
plot(y~x, col=color, pch=19)
year <- 2000+floor(x*10)
plot(y~year, col=ifelse(year>2007, cols[1], cols[2]), pch=19)
