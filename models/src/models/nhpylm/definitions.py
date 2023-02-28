# Beginning of chant
# Since 0 and 1 are not Char/String in Julia, might be useful to add some unique representations as well.
# Use alpha and omega then.
BOS_CHAR = 'Α'
EOS_CHAR = 'Ω'

# Since they are technically words, we should also store their hashed representations.
# Might well hash the string versions of them to ensure that their hash values don't conflict with those of the actual words.
BOS = hash(str(BOS_CHAR))
EOS = hash(str(EOS_CHAR))

# Just realized that these are probably not valid chars. Will need to change to some sort of char type
# This should be fine? AFAIK the text in the corpora are all full-width. Let's see.
# Lower-case alpha and omega
BOW = 'α'
EOW = 'ω'

HPYLM_INITIAL_D = 0.5
HPYLM_INITIAL_THETA = 2.0
# For hyperparameter sampling as shown in Teh technical report expression (40)
HPYLM_A = 1.0
HPYLM_B = 1.0
# For hyperparameter sampling as shown in Teh technical report expression (40)
HPYLM_ALPHA = 1.0
HPYLM_BETA = 1.0

# const CHPYLM_β_STOP = 4.0
# const CHPYLM_β_PASS = 4.0
# The apparent best values taken from the paper.
CHPYLM_BETA_STOP = 0.57
CHPYLM_BETA_PASS = 0.85
CHPYLM_EPSILON = 1e-12