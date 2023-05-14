from gabc2volpiano.converter import VolpianoConverter
import pandas as pd
import csv


def gabc2chantstrings(gabc_chants, modes):
    """
    Convert all chants from gabc to volpiano format using gabc2volpiano project.
    If the project gabc2volpiano is not installed, use the function src.utils.importers.download_gabc2volpiano.
    Convert the volpiano format to chant melody, remove or '-' and other extra characters.
    Replace phrases symbols ('3', '4', '6', '7') by '|'.
    Print mode distribution of final phrased chants.

    Parameters
    ----------
    gabc_chants : list of strings
        list of chant melodies in gabc format
    modes : list of strings
        list of modes of gabc chants
    Returns
    -------
    chants : list of strings
        list of chant melodies containing '|' at the end of phrases
    """
    converter = VolpianoConverter()
    chants = []
    mode_counts = {}
    for gabc, mode in zip(gabc_chants, modes):
        try:
            _, volpiano = converter.convert(gabc)
            if '~' in volpiano:
                print("~")
            chant_string = volpiano.replace("-", "")
            chant_string = chant_string.replace(".", "")
            chant_string = chant_string.replace(" ", "")
            chant_string = chant_string.replace("1", "")
            chant_string = chant_string.replace("3", "|") # Pausa Major - Barline
            chant_string = chant_string.replace("4", "|") # Pausa Finalis - Double Barline
            chant_string = chant_string.replace("6", "|") # Pausa Minor - Breath Mark
            chant_string = chant_string.replace("7", "|") # Pausa Minima - Breath Mark


            if len(chant_string) > 0:  
                chants.append(chant_string)
                if not mode in mode_counts:
                    mode_counts[mode] = 0
                mode_counts[mode] += 1
            else:
                raise Exception("Chant has no melody..")
        except Exception as e:
            print("There was a converter error.", e)
    print("Phrased chant's mode distribution: ", mode_counts)
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
    modes = []
    for gabc, office_part, mode in zip(pd_gabc["gabc"], pd_gabc["office_part"], pd_gabc["mode"]):
        if genre == office_part:
            gabc_chants.append(gabc)
            modes.append(mode)
    chant_strings = gabc2chantstrings(gabc_chants=gabc_chants, modes=modes)
    rows = []
    for chant in chant_strings:
        rows.append([chant])
    # field names
    fields = ['chant_strings']

    with open(chant_strings_file, 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(rows)