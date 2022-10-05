def get_opt():
    opt = {
        'dataset':'',
        'loss':'',
        'optimizer':'',
        'model':'',
    }

    return opt



if __name__=='__main__':
    opt = get_opt()
    print(opt)