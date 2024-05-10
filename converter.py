import pandas as pd

def rw_to_fh(txt_file_path):
    # Convert rw file in fh-format
    with open(txt_file_path) as f:
        df_RW = pd.read_csv(f, sep="\t")
        fh_text = ''
        print(df_RW)
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