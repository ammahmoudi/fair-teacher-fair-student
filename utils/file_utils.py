
import shutil


def del_files(dir_path):
    shutil.rmtree(dir_path)



def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content

def load_txt_content(txt_path):
  
    with open(txt_path, 'r') as f:
        content = f.read()
    return content
