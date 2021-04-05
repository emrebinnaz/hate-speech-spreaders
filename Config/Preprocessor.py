import preprocessor as preprocessor

def getPreprocessor():

    preprocessor.set_options(preprocessor.OPT.MENTION,
                             preprocessor.OPT.URL,
                             preprocessor.OPT.RESERVED,
                             preprocessor.OPT.EMOJI,
                             preprocessor.OPT.SMILEY)

    return preprocessor