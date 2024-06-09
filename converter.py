import pandas as pd
from math import floor

def rw_to_fh(txt_file_path: str):
    # Convert rw file in fh-format
    with open(txt_file_path) as f:
        df_RW = pd.read_csv(f, sep="\t")
        fh_text = ''
        for col in df_RW.columns:
            list_strings = list(filter(lambda x: str(x)!='nan', df_RW[col].tolist()))
            list_strings = [str(int(n)) for n in list_strings]
            rw_str = '\n'.join(list_strings)
            text = '\n'.join(['\n\nHEADER:',
            'DataFormat=Tree',
            f'KeyCode={col}',
            'Unit=1/1000 mm',
            f'Length={len(list_strings)}',
            f'DateBegin={2022 - len(list_strings)+1}',
            'DateEnd=2022',
            'Dated=Dated',
            'DATA:Single',
            f'{rw_str}'])
            fh_text += text
            # print(text)
        f.close()
    with open(f'shrubs/arst/{col}.fh', 'w') as f:
        f.write(fh_text)

def rw_to_rwl(txt_file_path: str, end_year: int=2022):
    # Convert rw file in rwl-format
    with open(txt_file_path) as f:
        df_RW = pd.read_csv(f, sep="\t")
        rwl_text = ''
        for col in df_RW.columns[1:].sort_values():
            list_strings = list(filter(lambda x: str(x)!='nan', df_RW[col].tolist()))
            list_strings = [str(int(n)) for n in list_strings]
            
            start_year = end_year - len(list_strings) + 1
            end_dec_year = end_year
            start_dec_year = int(floor(end_dec_year / 10) * 10)
            
            while list_strings:
                year_str = ''
                for y in range(start_dec_year, end_dec_year+1):
                    new_year = list_strings.pop(0)
                    new_year = ' ' * (6 - len(new_year)) + new_year
                    year_str = new_year + year_str

                new_str = f'{col.replace(" ", "")}' + ' ' * (8-len(col.replace(" ", ""))) + f'{start_dec_year}{year_str}'
                if end_dec_year == end_year:
                    new_str += ' -9999'
                new_str += '\n'
                rwl_text = new_str + rwl_text
                # print(rwl_text)

                end_dec_year = start_dec_year - 1
                start_dec_year -= 10
                if start_dec_year < start_year:
                    start_dec_year = start_year
    with open('D:/learn/results_PrPa/mxpor.rwl', 'w') as f:
        f.write(rwl_text)
        f.close()