"""
===============================================
Notes:
 - implemented distributions that are not listed below are: equal, random
 - if a specified subgroup attribute (e.g., sex) is provided as an attribute (args.attributes)
    but is not mentioned in the selected distribution, equal distribution will be assumed for that subgroup
 - if a specifc attribute option is not mentioned (e.g., Female), then that specific attribute option will not be included (same as a value of 0)
 - to specify a distribution based on a combination of subgroup attributes (e.g., Female-Negative), list the custom distribution in 
    the "Specific Combinations" section of the distributions dictionary, and declare the name of the distribution as normal.
 - To see a list of supported abbreviations, see attribute_abbreviations at the bottom of this file
===============================================
Below are template dictionaries for different patient and image attributes, to be copy pasted and combined in the 'distributions' variable

'sex':{'M':1,'F':1},
'race':{'W':1,'B':1},
'COVID_status':{'P':1, 'N':1},
"""
# TODO - Update instructions!

distributions = {
    "100Male":{'M':1,'F':0},
    "100Female":{'M':0,'F':1},
    "MP-FN":{"MP":1,'FN':1,'MN':0,'FP':0},
    "BP-WN":{'BP':1, 'BN':0, 'WP':0, 'WN':1},
    "equal-binary":{"FBN":1, "FBP":1, "FWN":1, "FWP":1, "MBN":1, "MBP":1, "MWN":1, "MWP":1}
    
}

attribute_abbreviations = {
    "White":"W",
    "Black or African American":"B",
    "Female":'F',
    'Male':'M',
    "Yes":"P", # COVID+
    "No":'N', # COVID-
}
