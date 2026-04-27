import argparse
import os

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--kerberos", type=str, required=True)

    return parser.parse_args()

def main():
    
    args = parse()

    train_file = "train_"+args.dataset+".py"


    os.system(f"python {train_file} --data_dir {args.data_dir} --model_dir {args.model_dir} --kerberos {args.kerberos}")


if __name__ == "__main__":
    main()