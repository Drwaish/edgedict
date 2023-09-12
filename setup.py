import shutil

def main(source_path):
    src_path = source_path
    dst_path = "/"
    shutil.move(src_path, dst_path) 

if __name__ == "__main__":
    src_path = input("Enter Path of Log File")
    main(src_path)
