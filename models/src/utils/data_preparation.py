from gabc2volpiano.converter import VolpianoConverter
import pandas as pd
import csv


def gabc2chantstrings(gabc_chants):
    """
    Convert all chants from gabc to volpiano format using gabc2volpiano project.
    If the project gabc2volpiano is not installed, use the function src.utils.importers.download_gabc2volpiano.
    Convert the volpiano format to chant melody, remove or '-' and other extra characters.
    Replace phrases symbols ('7') by '|'.

    Parameters
    ----------
    gabc_chants : list of strings
        list of chant melodies in gabc format
    Returns
    -------
    chants : list of strings
        list of chant melodies containing '|' at the end of phrases
    """
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

def convert_gabc_to_chantstring_csv(gabc_csv_file = "gabc-chants.csv", 
                                    chant_strings_file = "gregobase-chantstrings-an.csv", 
                                    genre = "an"):
    """
    Load gregobase chants.csv file, process all chant melodies in gabc format and store results into 
    gregobase-chantstrings csv file for the specific genre. The new csv file contains melodies with
    '|' symbol that symbolizes end of phrase, e.g. "adkdjjjkk|kkjkjjjpaabjkj|asdas|aasdasdasdsadaggads".

    Parameters
    ----------
    gabc_csv_file : str
        path to gabc-chants.csv file, chants.csv file from gregocorpus
    chant_strings_file : str
        path to new csv file of preprocessed gregobase chant melodies separated by phrases (using symbol '|')
    genre : str
        genre type, 'an' stands for antiphona, 're' stands for responsorium
    """
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