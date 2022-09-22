from args import args_parser
from Server import Server_

def main():
    args = args_parser()
    Server = Server_(args)
    Server.server()
    Server.global_test()

if __name__ == '__main__':
    main()
