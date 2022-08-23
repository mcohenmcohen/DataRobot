library(data.table)
dat <- fread('~/datasets/jp_readmission.csv')
dat2 <- fread('~/datasets/dirty_simulated_1k.csv')

text_to_fixture <- function(x, name){
  #x <- head(x, 1000)
  x <- paste0("    '", x, "'")
  x <- paste(x, collapse=",\n")
  x <- paste0(name, " = [\n", x, "\n]")
  return(x)
}

diag_1 <- text_to_fixture(dat$diag_1_desc_診断1説明, 'jp_diag_1')
diag_2 <- text_to_fixture(dat$diag_2_desc_診断2説明, 'jp_diag_2')
diag_3 <- text_to_fixture(dat$diag_3_desc_診断3説明, 'jp_diag_3')

high_card1 <- text_to_fixture(dat2$NoiseHighCard1, 'high_card_1')
high_card2 <- text_to_fixture(dat2$NoiseHighCard2, 'high_card_2')
high_card3 <- text_to_fixture(dat2$NoiseHighCard3, 'high_card_3')

out <- paste(
  diag_1, diag_2, diag_3,
  high_card1, high_card2, high_card3,
  sep='\n\n')

header <- "from __future__ import unicode_literals\n\n"
header <- paste0("# encoding: utf-8\n", header)
#header <- paste0("# pylint: disable=all", header)

out <- paste0(header, out, '\n')
#cat(out)

fname <- '~/workspace/DataRobot/tests/ModelingMachine/jp_text_fixtures.py'
unlink(fname)
cat(out, file=fname)
