library(data.table)
library(stringi)
x <- fread("http://id.loc.gov/vocabulary/iso639-1.tsv")
setnames(x, make.names(names(x)))

l <- c("af", "am", "an", "ar", "as", "az", "be", "bg", "bn", "br", "bs", "ca", "cs", "cy", "da", "de", "dz", "el", "en", "eo", "es", "et", "eu", "fa", "fi", "fo", "fr", "ga", "gl", "gu", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "ku", "ky", "la", "lb", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "mt", "nb", "ne", "nl", "nn", "no", "oc", "or", "pa", "pl", "ps", "pt", "qu", "ro", "ru", "rw", "se", "si", "sk", "sl", "sq", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "ug", "uk", "ur", "vi", "vo", "wa", "xh", "zh", "zu")

#x <- x[code %in% l, c("code", "Label..English."),with=FALSE]
x <- x[, c("code", "Label..English."),with=FALSE]
setnames(x, c('code', 'lang'))
x[,lang := tolower(lang)]
x[,lang := sapply(strsplit(lang, "|", fixed=TRUE), "[", 1)]
x[,lang := stri_trim(lang)]


x[,code := paste0("u'", code, "'")]
x[,lang := paste0("u'", lang, "'")]

#Language to code
cat(x[,paste0("{", paste(paste0("\t", lang, ": ", code), collapse=",\n"), "}")])
