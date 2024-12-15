class SCANDataLoader:
    def __init__(self):
        pass

    def load_file_path(self, file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            src, tgt = line.strip().split(" OUT: ")
            src = src.replace("IN: ", "").split()
            tgt = tgt.split()
            data.append({"src": src, "tgt": tgt})
        return data
    
    def load_file_paths(self, file_paths):
        data = []
        for file_path in file_paths:
            data += self.load_file_path(file_path)
        return data
