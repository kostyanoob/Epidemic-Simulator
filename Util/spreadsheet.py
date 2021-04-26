from shutil import copyfile
from collections import OrderedDict
import numpy as np
import os
import pandas as pd
from MyDate import MyDate
from Institution import Institution
from typing import Tuple, List, Union
import ezodf
import xlsxwriter
import openpyxl


def validate_main_spreadsheet(spreadsheet_path: str) -> Tuple[bool, str]:
    """
    Makes sure that the path to spreadsheet is valid.
    :param spreadsheet_path: string
    :return: True if the path seems valid, and if the path is
    invalid a string with a message will explain the reason
    """
    if spreadsheet_path is None:
        msg = "Invalid path to spreadsheet filename (None)."
        print(msg)
        return False, msg
    if len(spreadsheet_path) < 3:
        msg = "The spreadsheet filename ({}) is too short. "\
              "Must contain a proper filename extension.".format(
                  spreadsheet_path)
        print(msg)
        return False, msg
    if not os.path.exists(spreadsheet_path):
        msg = "The spreadsheet {} doesn't exist.".format(spreadsheet_path)
        print(msg)
        return False, msg
    return True, ""


def read_checklist_spreadsheet(spreadsheet_path: str) -> Union[pd.DataFrame, None]:
    """
    Reads an "ods" or an "xlsx" file and returns 1 pandas-dataframe
    containing only the people that were checked with a "v" sign
    :param spreadsheet_path:
    :return: a pandas Dataframe containing the checklist spreadsheet
    """
    value_list = [
        "V", "v", 1]  # these are the symbols that will be regarded as "checked=True" in the "Tested" column
    if len(spreadsheet_path) < 3:
        print("The filename is too short. Must contain a proper filename extension.")
        return None
    elif spreadsheet_path[-3:] == "ods":
        df = read_ods(filename=spreadsheet_path, sheet=0)
        df = set_checklist_dataframe_column_types(df)
        return df[df.Tested.isin(value_list)]
    elif len(spreadsheet_path) >= 4 and spreadsheet_path[-4:] == "xlsx":
        df = read_excel(spreadsheet_path, skip_blank_lines=True, sheet=0)
        df = set_checklist_dataframe_column_types(df)
        return df[df.Tested.isin(value_list)]
    else:
        raise FileNotFoundError(
            "This type of file cannot be handled at this time.")


def stringify_cell(cell_value):
    """attempts to cast the cell_value to a string"""
    if pd.isnull(cell_value):
        str_value = cell_value
    else:
        try:
            str_value = str(int(float(cell_value)))
        except ValueError:
            str_value = str(cell_value)
    return str_value


def set_organization_dataframe_column_types(organization_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sets the first 4 columns to be strings, and the rest of the columns to be integers
    :param organization_df:
    :return:
    """
    type_lst = [str] * 4 + [int] * (len(organization_df.columns)-4)
    convert_dict = {colname: typename for colname,
                    typename in zip(organization_df.columns, type_lst)}
    organization_df = organization_df.astype(convert_dict)
    organization_df[organization_df.columns[0]
                    ] = organization_df[organization_df.columns[0]].apply(stringify_cell)
    organization_df[organization_df.columns[1]
                    ] = organization_df[organization_df.columns[1]].apply(stringify_cell)
    return organization_df


def set_delta_datafame_column_types(delta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sets the first 3 columns to be strings, and the rest of the columns to be floats
    :param delta_df:
    :return:
    """
    type_lst = [str] * 3 + [float] + [str]
    convert_dict = {colname: typename for colname,
                    typename in zip(delta_df.columns, type_lst)}
    delta_df = delta_df.astype(convert_dict)
    delta_df[delta_df.columns[0]
             ] = delta_df[delta_df.columns[0]].apply(stringify_cell)
    delta_df[delta_df.columns[1]
             ] = delta_df[delta_df.columns[1]].apply(stringify_cell)
    return delta_df


def set_ext_risk_dataframe_column_types(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sets the first 5 columns to be strings, and the rest of the columns to be integers
    :param risk_df:
    :return:
    """
    # using dictionary to convert specific columns
    type_lst = [str] * 5 + [float] * (len(risk_df.columns) - 5)
    convert_dict = {colname: typename for colname,
                    typename in zip(risk_df.columns, type_lst)}
    risk_df = risk_df.astype(convert_dict)
    risk_df[risk_df.columns[0]] = risk_df[risk_df.columns[0]].apply(
        stringify_cell)
    risk_df[risk_df.columns[1]] = risk_df[risk_df.columns[1]].apply(
        stringify_cell)
    return risk_df


def set_int_risk_dataframe_column_types(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sets the first 3 columns to be strings, and the rest of the columns to be integers
    :param risk_df:
    :return:
    """
    type_lst = [str] * 3 + [float] * (len(risk_df.columns) - 3)
    convert_dict = {colname: typename for colname,
                    typename in zip(risk_df.columns, type_lst)}
    risk_df = risk_df.astype(convert_dict)
    risk_df[risk_df.columns[0]] = risk_df[risk_df.columns[0]].apply(
        stringify_cell)
    risk_df[risk_df.columns[1]] = risk_df[risk_df.columns[1]].apply(
        stringify_cell)
    return risk_df


def set_other_dataframe_column_types(other_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sets the first 2 columns to be strings, and the rest of the columns to be integers
    :param other_df:
    :return:
    """
    type_lst = [str] * 2 + [float] * (len(other_df.columns) - 2)
    convert_dict = {colname: typename for colname,
                    typename in zip(other_df.columns, type_lst)}
    other_df = other_df.astype(convert_dict)
    other_df[other_df.columns[0]] = other_df[other_df.columns[0]].apply(
        stringify_cell)
    other_df[other_df.columns[1]] = other_df[other_df.columns[1]].apply(
        stringify_cell)
    return other_df


def set_checklist_dataframe_column_types(checklist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sets all the columns to the type string.
    :param checklist_df:
    :return:
    """

    convert_dict = {colname: str for colname in checklist_df.columns}
    checklist_df = checklist_df.astype(convert_dict)
    checklist_df[checklist_df.columns[0]
                 ] = checklist_df[checklist_df.columns[0]].apply(stringify_cell)
    checklist_df[checklist_df.columns[1]
                 ] = checklist_df[checklist_df.columns[1]].apply(stringify_cell)
    return checklist_df


def read_main_spreadsheet(spreadsheet_path: str) -> Tuple[Union[pd.DataFrame, None], Union[pd.DataFrame, None], Union[pd.DataFrame, None], Union[pd.DataFrame, None], Union[pd.DataFrame, None], str]:
    """
    Reads an "ods" or an "xlsx" file and returns 3 pandas-dataframes with the corresponding sheets in them.
    Replaces empty values with zeroes.
    :param spreadsheet_path:
    :return:
    """
    path_valid, path_err_msg = validate_main_spreadsheet(spreadsheet_path)
    if not path_valid:
        return None, None, None, None, None, path_err_msg
    elif len(spreadsheet_path) < 4 or (spreadsheet_path[-4:] != "xlsx" and spreadsheet_path[-3:] != "ods"):
        err_msg = "This type of spreadsheet cannot be handled at this time. Only ods and xlsx are supported."
        print(err_msg)
        return None, None, None, None, None, err_msg
    elif spreadsheet_path[-3:] == "ods":
        organization_df = read_ods(filename=spreadsheet_path, sheet=0)
        ext_risk_df = read_ods(filename=spreadsheet_path, sheet=1)
        int_risk_df = read_ods(filename=spreadsheet_path, sheet=2)
        group_risk_df = read_ods(filename=spreadsheet_path, sheet=3)
        delta_df = read_ods(filename=spreadsheet_path, sheet=4)
    elif spreadsheet_path[-4:] == "xlsx":
        organization_df = read_excel(
            spreadsheet_path, skip_blank_lines=True, sheet=0)
        ext_risk_df = read_excel(
            spreadsheet_path, skip_blank_lines=True, sheet=1)
        int_risk_df = read_excel(
            spreadsheet_path, skip_blank_lines=True, sheet=2)
        group_risk_df = read_excel(
            spreadsheet_path, skip_blank_lines=True, sheet=3)
        delta_df = read_excel(spreadsheet_path, skip_blank_lines=True, sheet=4)
    else:
        err_msg = "This type of spreadsheet cannot be handled at this time. Only ods and xlsx are supported."
        print(err_msg)
        return None, None, None, None, None, err_msg
    organization_df = set_organization_dataframe_column_types(organization_df)
    ext_risk_df = set_ext_risk_dataframe_column_types(ext_risk_df)
    group_risk_df = set_other_dataframe_column_types(group_risk_df)
    int_risk_df = set_int_risk_dataframe_column_types(
        int_risk_df)
    delta_df = set_delta_datafame_column_types(delta_df)

    # Sanity checks
    if organization_df.shape[0] != ext_risk_df.shape[0] or organization_df.shape[0] != delta_df.shape[0] or organization_df.shape[0] != int_risk_df.shape[0]:
        err_msg = "Error: the sheets 'Organization','Ext_Risk', 'Delta' and 'Int_Risk' of the {} "\
            "spreadsheet do not agree on the number of people.".format(
                spreadsheet_path)
        print(err_msg)
        return None, None, None, None, None, err_msg
    if not organization_df[organization_df.columns[:3]].equals(ext_risk_df[ext_risk_df.columns[:3]]) or not organization_df[organization_df.columns[:3]].equals(delta_df[delta_df.columns[:3]]) or not organization_df[organization_df.columns[:3]].equals(int_risk_df[int_risk_df.columns[:3]]):
        err_msg = "Error: the sheets 'Organization','Ext_Risk', 'Delta' and 'Int_Risk'of the {} "\
                  "spreadsheet have differences in the name lists.".format(
                      spreadsheet_path)
        print(err_msg)
        return None, None, None, None, None, err_msg
    return organization_df, ext_risk_df, group_risk_df, int_risk_df, delta_df, ""


def read_excel(filename: str, skip_blank_lines: bool, sheet: int, usecols=None) -> pd.DataFrame:
    """
    reads an excel file and returns a pandas dataframe
    :param filename: a path to the excel file.
    :param skip_blank_lines: if True, then blank lines won't be included in the returned
                             dataframe
    :param sheet: integer, the serial number of the sheet to be read in the spreadsheet
    :param usecols: a list of column indices to keep for the returned dataframe.
                    If None, then all the columns will be returned
    :return: a dataframe, obtained from a single sheet of the excel filename.
    """
    df = pd.read_excel(filename, skip_blank_lines=skip_blank_lines,
                       sheet_name=sheet, usecols=usecols)
    df.dropna(axis=0, how='all', thresh=None, subset=None, inplace=True)
    df = df.fillna(0)
    df.dropna(axis=1, how='all', thresh=None, subset=None, inplace=True)

    # replace field that's entirely space (or empty) with NaN
    df = df.replace(r'^\s*$', 0.0, regex=True)
    return df


def read_ods(filename, sheet=0, header=0, usecols=None) -> pd.DataFrame:
    """
    Reads an "ods" file and returns a pandas-dataframe. Replaces empty values with zeroes.
    :param filename: a path to the ods file.
    :param sheet: integer, the serial number of the sheet to be read in the spreadsheet
    :param header: integer, the row number that contains the headers of the columns
    :param usecols: list of column indices to be kept for the returned dataframe
    :return: a dataframe, obtained from a single sheet of the excel filename.
    """
    tab = ezodf.opendoc(filename=filename).sheets[sheet]
    if usecols is None:
        df = pd.DataFrame(
            {col[header].value: [x.value for x in col[header + 1:]] for col in tab.columns()})
    else:
        dict_for_df = {}
        for col_idx, col in enumerate(tab.columns()):
            if col_idx in usecols:
                dict_for_df[col[header].value] = [
                    x.value for x in col[header + 1:]]
        df = pd.DataFrame(dict_for_df)
    df = df.replace(to_replace='None', value=np.nan)
    df.dropna(axis=0, how='all', thresh=None, subset=None, inplace=True)
    df = df.fillna(0)
    df.dropna(axis=1, how='all', thresh=None, subset=None, inplace=True)
    # replace field that's entirely space (or empty) with NaN
    df = df.replace(r'^\s*$', 0.0, regex=True)
    return df


def write_new_spreadsheet_to_file(dataframes: List[pd.DataFrame], sheet_names: List[str],
                                  path_to_new_file: str, wide_columns: bool = True) -> Tuple[bool, str]:
    """
    Writes a 2-sheet spreadsheet to a file. The type of the spreadsheet is
    determined by the las characters in path_to_new_file.
    :param dataframes: list of pandas dataframes tobe written to the sheets in the spreadsheet file.
    :param sheet_names: list of string, which will determine the sheet names of in the spreadsheet file.
    :param path_to_new_file: the full path to a spreadsheet file to be created.
    :param wide_columns: True if in the written spreadsheet, the columns should be wide
    :return: a 2-tuple: (True <---> succeeded, message)
    """
    if path_to_new_file[-4:] == "xlsx":
        writer = pd.ExcelWriter(path_to_new_file, engine='xlsxwriter')
        for df, sheet_name in zip(dataframes, sheet_names):
            df.to_excel(excel_writer=writer, sheet_name=sheet_name,
                        header=True, index=False)
            if wide_columns:
                for column in df.columns:
                    column_length = max(df[column].astype(
                        str).map(len).max(), len(column))
                    col_idx = df.columns.get_loc(column)
                    writer.sheets[sheet_name].set_column(
                        col_idx, col_idx, column_length)
        writer.save()
        writer.close()
        msg = "\nCreated/modify the excel file {}".format(path_to_new_file)
        print(msg)
        return True, msg
    elif path_to_new_file[-3:] == "ods":
        dict_of_sheets = OrderedDict()
        for df, sheet_name in zip(dataframes, sheet_names):
            # Initialize data to be written as an empty list, as pyods needs a list to write
            # Write the columns first to be the first entries
            current_sheet_row_wise = [list(df.columns)]
            # loop through data frame (row-wise) and update the data list
            for index, row in df.iterrows():
                current_sheet_row_wise.append(list(row.values))
            # Populate dict() with the sheet data
            dict_of_sheets.update({sheet_name: current_sheet_row_wise})

        # Finally call save_data() from pyods to store the ods file
        from pyexcel_ods import save_data as save_ods_data
        save_ods_data(path_to_new_file, dict_of_sheets)
        msg = "Created the ODS file {}".format(path_to_new_file)
        print(msg)
        return True, msg
    else:
        msg = "Warning the file {} was not created!".format(path_to_new_file)
        print(msg)
        return False, msg


def edit_speadsheet_xlsx(dataframes: List[pd.DataFrame], sheet_names: List[str],
                         path_to_new_file: str) -> Tuple[bool, str]:
    """
    Writes a 2-sheet spreadsheet to a file. The type of the spreadsheet is
    determined by the las characters in path_to_new_file.
    :param dataframes: list of pandas dataframes tobe written to the sheets in the spreadsheet file.
    :param sheet_names: list of string, which will determine the sheet names of in the spreadsheet file.
    :param path_to_new_file: the full path to a spreadsheet file to be created.
    :return: a 2-tuple: (True <---> succeeded, message)
    """
    assert path_to_new_file[-4:] == "xlsx", "The spreadsheet must be an .xlsx file"
    book = openpyxl.load_workbook(path_to_new_file)
    writer = pd.ExcelWriter(path_to_new_file, engine='openpyxl')
    writer.book = book
    for df, sheet_name in zip(dataframes, sheet_names):
        df.to_excel(excel_writer=writer, sheet_name=sheet_name,
                    header=True, index=False)
    writer.save()
    writer.close()
    msg = "\nCreated the excel file {}".format(path_to_new_file)
    print(msg)
    return True, msg


def produce_weighted_risk_sheet(organization_df: pd.DataFrame, risk_df: pd.DataFrame,
                                path_to_spreadsheet: str, institution: Institution):
    """
    Overwrites the spreadsheet specified by the :path_to_spreadsheet" argument.
    In the new file, 3 sheets will reside:
     1st sheet "Organization" - created by dumping the organization_df to it.
     2nd sheet "Risk" - created by dumping the risk_df to it.
     3rd sheet "Weighted Risk" - created by taking the risk_df, scaling the risk factor scores in it
                                 by the appropriate coefficient. Also 3 new columns will be added,
                                 containing the weighted sum of the risk factors, a discount factor
                                  and finally a weighted and a discounted risk of each person.

    :param organization_df: pandas dataframe representing the 1st sheet
    :param risk_df: pandas dataframe representing the 2nd sheet
    :param path_to_spreadsheet: a full path to a spreadsheet to be created.
    :param institution: Institution instance that contains information regarding the
                        weighted and the discounted risk factors of each person.
    """
    weighted_risk_df = risk_df.copy()
    rows, cols = weighted_risk_df.shape[0], weighted_risk_df.shape[1] + 3
    for column in weighted_risk_df.columns[institution.num_risk_df_columns_that_arent_risk_factors:]:
        # important to change the type of the factor column to float, otherwise floats here will be rounded!
        weighted_risk_df.loc[:, column] = weighted_risk_df[column].apply(
            lambda x: float(x))
    weighted_risk_df.insert(cols - 3, "Weighted Sum (risk)", rows*[0.0], False)
    weighted_risk_df.insert(
        cols - 2, "Discount Factor (lambda)", rows * [0.0], False)
    weighted_risk_df.insert(
        cols - 1, "Weighted and Discounted Sum (w=risk*lambda)", rows * [0.0], False)
    # Iterate the organization dataframe, construct the person keys (by concatenating 4 personal details)
    # Using the person keys, retrieve their information directly from the institution instance.
    # Set the new information to the weighted_risk_df.
    for rowid, organization_row in organization_df.iterrows():
        person = "{}_{}_{}_{}".format(str(organization_row[0]), str(
            organization_row[1]), str(organization_row[2]), str(organization_row[3]))
        weighted_risk_list = institution.nodes_attributes[person]['weighted_risk_vector'].tolist(
        )
        weighted_sum = institution.nodes_attributes[person]['risk_ext']
        discount_factor = institution.nodes_attributes[person]['discount_factor']
        weighted_and_discounted_sum = institution.nodes_attributes[person]['ext_infection_prob']
        new_data_list = weighted_risk_list + \
            [weighted_sum, discount_factor, weighted_and_discounted_sum]
        for colid, new_data in zip(range(institution.num_risk_df_columns_that_arent_risk_factors, cols), new_data_list):
            weighted_risk_df.at[rowid,
                                weighted_risk_df.columns[colid]] = new_data

    # Write weighted_risk_df as an additional sheet, alongside with the "organization_df" and the "risk_df" sheets
    try:
        write_succeeded, write_msg = write_new_spreadsheet_to_file([organization_df, risk_df, weighted_risk_df],
                                                                   ["Organization", "Risk",
                                                                       "Weighted Risk"],
                                                                   path_to_spreadsheet)

    except (xlsxwriter.exceptions.FileCreateError, PermissionError) as e:
        write_succeeded, write_msg = False, "Couldn't write the \"Weighted Risk\" sheet to the main " \
                                            "spreadsheet {} due to lack of permissions. " \
                                            "It may be open by some other program. Try closing applications " \
                                            "that may use the target spreadsheet, then retry.".format(
                                                path_to_spreadsheet)
    except IOError:
        write_succeeded, write_msg = False, "Failed adding a \"Weighted Risk\" sheet to the main " \
                                            "spreadsheet {}".format(
                                                path_to_spreadsheet)

    if write_succeeded:
        return True, ""
    else:
        return False, write_msg


def merge_checklist_to_main(checklist_path: str, main_path: str, newmain_path: str) -> Tuple[bool, str]:
    """
    The merging of "checklist_path" spreadsheet into "main_path" spreadsheet stands
    for updating the disease testing time of each person in the "main_path"
    spreadsheet to be the date of the checklist file. The resulting main spreadsheet
    is saved as a new file named "newmain_path".

    Note: If the checklist contains tested (marked with V) entries (people) that do
     not appear in the main file, these entries will be ignored. Namely, these
     entries will not be added to the newly created main spreadsheet (newmain_path)

    :param checklist_path:
    :param main_path:
    :param newmain_path:
    :return: True If at least the reading of the existing spreadsheets succeeded.
    At this point this function cannot indicate whether the writing of the new
    spreadsheet succeeded.
    """
    organization_df, ext_risk_df, group_risk_df, int_risk_df, delta_df = read_main_spreadsheet(
        main_path)
    if errmsg != "":
        assert organization_df is None and risk_df is None
        return False, errmsg
    # containing only people that were checked
    checklist_df = read_checklist_spreadsheet(checklist_path)
    if checklist_df is None:
        return False, "Failed loading the checklist from " + checklist_path
    else:
        assert checklist_df is not None and organization_df is not None and int_risk_df is not None and ext_risk_df is not None and delta_df is not None and group_risk_df is not None
        checklist_date = MyDate(strdate=checklist_path.split(
            os.path.sep)[-1].split("_")[0])
        suspicious_checklist_entries_df = checklist_df.loc[~checklist_df[checklist_df.columns[0]].isin(
            risk_df[risk_df.columns[0]])]  # appear in checklist but not in the main df
        errmsg = ""
        if len(suspicious_checklist_entries_df) > 0:
            errmsg = "Warning: the checklist {} contained the following entries that do not "\
                "appear in the main file {}. These entries were NOT added to {}. If you want " \
                "them to be included in the main file, you need to remove {}, add " \
                     "these people to {} and then re-run this program. " \
                     "\n".format(checklist_path, main_path,
                                 newmain_path, newmain_path, main_path)
            errmsg += str(suspicious_checklist_entries_df)

        # Update the people that appear both in the risk_df and in the checklist_df to have the checklist's test date
        covid_test_col_str = 'Date of last COVID19 test' if 'Date of last COVID19 test' in ext_risk_df.columns else 'תאריך בדיקה אחרון'
        ext_risk_df.loc[ext_risk_df[ext_risk_df.columns[0]].isin(
            checklist_df[checklist_df.columns[0]]), covid_test_col_str] = checklist_date.strdate
        try:
            write_succeeded, write_msg = write_new_spreadsheet_to_file([organization_df, ext_risk_df, group_risk_df, int_risk_df, delta_df],
                                                                       ["Organization", "External Risk", "Internal Risk", "Group Risk", "Delta_init"],
                                                                       newmain_path)
        except PermissionError:
            write_succeeded, write_msg = False, "Couldn't create the main spreadsheet due to lack of permission. " \
                                                "It may be open by some other program. Try closing applications " \
                                                "that may use the target spreadsheet, then retry."
        except IOError:
            write_succeeded, write_msg = False, "Couldn't create the main spreadsheet. "

        if write_succeeded:
            return True, errmsg + '\n' + write_msg
        else:
            return False, write_msg


def set_up_main_spreadsheet(current_date: MyDate, previous_date: MyDate, spreadsheet_directory: str,
                            spreadsheet_filename_extension: str):
    """
    Assume we are using spreadsheet_filename_extension="xlsx":

    If <current_date>_main.xlsx exists : just report that you are about to use the current version of
                                         <current_date>_main.xlsx return True
    If <current_date>_main.xlsx doesn't exist :
        If <previous_date>_main.xlsx exists :
            If <previous_date>_checklist.xlsx exists : <current_date>_main.xlsx = <previous_date>_checklist.xlsx +
                                                                                  <previous_date>_main.xlsx
            If <previous_date>_checklist.xlsx doesn't exist : <current_date>_main.xlsx = <previous_date>_main.xlsx
        If <previous_date>_main.xlsx doesn't exist : "Error!" return False

    :param current_date: MyDate object
    :param previous_date: MyDate object
    :param spreadsheet_directory: str, a path to the directory that contains all the "main" and "checklist" spreadsheets
    :param spreadsheet_filename_extension: str, an extension (c.f. "xlsx" or "ods")

    :return: string with the full path to the new spreadsheet <--> <current_date>_main.xlsx is ready for use
             None <---> there was a problem setting up the main spreadsheet
    """
    current_main_filepath = os.path.join(spreadsheet_directory, "{}_main.{}".format(current_date.strdate,
                                                                                    spreadsheet_filename_extension))
    previous_main_filepath = os.path.join(spreadsheet_directory, "{}_main.{}".format(previous_date.strdate,
                                                                                     spreadsheet_filename_extension))
    previous_checklist_filepath = os.path.join(spreadsheet_directory,
                                               "{}_checklist.{}".format(previous_date.strdate,
                                                                        spreadsheet_filename_extension))

    if not os.path.exists(spreadsheet_directory):
        print("Error: directory {} doesn't exist. Make sure you create it and "
              "place an initial spreadsheet there.".format(spreadsheet_directory))
        return None
    elif os.path.exists(current_main_filepath):
        print("Found {}, solving optimum people selection based on this file solely.".format(
            current_main_filepath))
        return current_main_filepath
    elif os.path.exists(previous_main_filepath):
        if os.path.exists(previous_checklist_filepath):
            print("Merging {} into {} and saving the result as {}".format(previous_checklist_filepath,
                                                                          previous_main_filepath,
                                                                          current_main_filepath))
            merge_succeeded, _ = merge_checklist_to_main(checklist_path=previous_checklist_filepath,
                                                         main_path=previous_main_filepath,
                                                         newmain_path=current_main_filepath)
            return current_main_filepath if merge_succeeded else None
        else:
            print("While attempting to create the missing current main spreadsheet, the required file {} was not "
                  "found. Hence, the current main spreadsheet ({}) will be just an exact copy of "
                  "{}".format(previous_checklist_filepath, current_main_filepath, previous_main_filepath))
            copyfile(previous_main_filepath, current_main_filepath)
            return current_main_filepath
    else:  # the previous main spreadsheet doesn't exist
        print("Neither the current main spreadsheet ({}) nor the previous main spreadsheet ({})"
              " were found. Cannot proceed.".format(current_main_filepath, previous_main_filepath))
        return None


def produce_checklist(sampled_person_lst: list, current_date: MyDate, spreadsheet_directory: str,
                      spreadsheet_filename_extension: str) -> Tuple[bool, str]:
    """
    Creates a checklist spreadsheet with the people that should be tested today.
    :param sampled_person_lst:
    :param current_date:
    :param spreadsheet_directory:
    :param spreadsheet_filename_extension:
    :return:
    """
    #
    path = os.path.join(spreadsheet_directory, "{}_checklist.{}".format(
        current_date, spreadsheet_filename_extension))
    data = [person.split("_") + [""] for person in sampled_person_lst]
    df = pd.DataFrame(
        data, columns=['Worker ID', 'Citizen ID', 'Full Name', 'Position', 'Tested'])
    try:
        outcome, msg = write_new_spreadsheet_to_file([df], ['Checklist'], path)
    except PermissionError:
        outcome, msg = False, "Couldn't create the checklist file due to lack of permission. " \
                              "It may be open by some other program. Try closing applications " \
                              "that may use the target spreadsheet, then retry the checklist creation."
    except IOError:
        outcome, msg = False, "Couldn't create the checklist file. "

    return outcome, msg
