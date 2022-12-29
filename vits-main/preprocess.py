import argparse
import text
from utils import load_filepaths_and_text

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="cleaned")
  parser.add_argument("--text_index", default=1, type=int)
  parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
  parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])

  args = parser.parse_args()
    

  for filelist in args.filelists:
    print("START:", filelist)
    filepaths_and_text = load_filepaths_and_text(filelist)
    for i in range(len(filepaths_and_text)):
      original_text = filepaths_and_text[i][args.text_index]
      cleaned_text = text._clean_text(original_text, args.text_cleaners)
      filepaths_and_text[i][args.text_index] = cleaned_text

    new_filelist = filelist + "." + args.out_extension
    with open(new_filelist, "w", encoding="utf-8") as f:
      f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])

# 这段代码是一个命令行程序，用于将一个或多个文件列表中的文本进行清洗。它使用 argparse 模块处理命令行参数，并使用 load_filepaths_and_text 函数从文件列表中加载文件路径和文本。
# 然后，对于每个文件列表，程序将遍历其中的所有文本并使用 text._clean_text 函数将其清洗。清洗函数使用 text_cleaners 参数指定的清洗器（或清洗器列表）对文本进行处理，具体取决于 text_cleaners 参数的值。
# 最后，程序将清洗后的文本保存到新的文件列表中，新文件列表的文件名为原始文件列表名加上 out_extension 参数指定的扩展名。
# 例如，如果将 out_extension 设置为 "cleaned"，则新文件列表的文件名将为 "原始文件列表名.cleaned"。
