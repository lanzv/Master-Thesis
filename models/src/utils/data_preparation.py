from gabc2volpiano.converter import VolpianoConverter
import pandas as pd
import csv


def gabc2chantstrings(gabc_chants):
    converter = VolpianoConverter()
    chants = []
    for gabc in gabc_chants:
      try:
          _, volpiano = converter.convert(gabc)
          if '~' in volpiano:
              print("~")
          chant_string = volpiano.replace("-", "")
          chant_string = chant_string.replace(".", "")
          chant_string = chant_string.replace(" ", "")
          chant_string = chant_string.replace("1", "")
          chant_string = chant_string.replace("3", "") # Barline
          chant_string = chant_string.replace("4", "") # Double barline
          chant_string = chant_string.replace("6", "") # Middle barline
          chant_string = chant_string.replace("7", "|") # Commas
          if len(chant_string) > 0:
              chants.append(chant_string)
          else:
              raise Exception("Chant has no melody..")
      except Exception as e:
        print("There was a converter error.", e)

    return chants

def convert_gabc_to_chantstring_csv(gabc_csv_file = "gabc-chants.csv", chant_strings_file = "gregobase-chantstrings-an.csv", genre = "an"): # an stands for antiphona, re stands for responsorium
    pd_gabc = pd.read_csv(gabc_csv_file, index_col='id')
    gabc_chants = []
    for gabc, office_part in zip(pd_gabc["gabc"], pd_gabc["office_part"]):
        if genre == office_part:
            gabc_chants.append(gabc)
    chant_strings = gabc2chantstrings(gabc_chants=gabc_chants)
    rows = []
    for chant in chant_strings:
        rows.append([chant])
    # field names
    fields = ['chant_strings']

    with open(chant_strings_file, 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(rows)