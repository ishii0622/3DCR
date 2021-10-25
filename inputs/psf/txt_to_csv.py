import csv


# テキストファイルを開いてCSVファイルに書き込む
def text_csv_converter(datas):  # datasはテキストファイルの場所
    # 保存するCSVファイルの場所
    file_csv = datas.replace("txt", "csv")

# テキストファイルを開く
    with open(datas)as rf:
        # 書き込むＣＳＶファイルを開く
        with open(file_csv, "w")as wf:
            # テキストを１行ずつ読み込む
            # テキストの１行を要素としたlistになる
            readfile = rf.readlines()

            for read_text in readfile:
                # listに分割
                read_text = read_text.split()
                # csvに書き込む
                writer = csv.writer(wf, delimiter=',')
                writer.writerow(read_text)


if __name__ == '__main__':
    # filenameはテキストファイルの場所
    filename = "PSF.txt"
    text_csv_converter(filename)