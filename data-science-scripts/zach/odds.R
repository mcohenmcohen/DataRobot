# From https://electionbettingodds.com/
library(data.table)
general_election <- list(
  data.table(person='biden', odds=.147/.310),
  data.table(person='sanders', odds=.128/.292),
  data.table(person='warren', odds=.035/.112),
  data.table(person='bloomberg', odds=.057/.103),
  data.table(person='buttigieg', odds=.023/.068),
  data.table(person='clinton', odds=.012/.029),
  data.table(person='yang', odds=.017/.024)
)
general_election <- rbindlist(general_election, fill=T, use.names=T)
general_election <- general_election[order(odds, decreasing=T),]

actual_trump_odds_overall <- .542/.913
print(actual_trump_odds_overall)

print(general_election)