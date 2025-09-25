import pandas as pd
import numpy as np

class DataHandler:
    def __init__(self, filename):
        """Initialize the data handler and load x and y from the given file."""
        filetype = self._check_file_type(filename)

        if filetype == 'MEM':
            self.x, self.y, self.prescan_indices, self.postscan_indices = self._read_mem(filename)
        else:
            self.x, self.y = self._read_csv(filename)
        
        self.x_trimmed = self.x[self.prescan_indices[1]:self.postscan_indices[0]]
        self.y_trimmed = self.y[self.prescan_indices[1]:self.postscan_indices[0]]

        self.name = self._find_name(filename)

    def _check_file_type(self, filename):
        """Return the file type based on the extension."""
        if filename.lower().endswith('.mem'):
            return 'MEM'
        elif filename.lower().endswith('.csv'):
            return 'CSV'
        else:
            raise ValueError("Unsupported file type.")

    def _read_csv(self, filename):
        """Read data from CSV and return Stims and Amps."""
        data = pd.read_csv(filename)
        stims = data["S"].to_numpy().ravel()
        amps = data["A"].to_numpy().ravel()
        return stims, amps

    def _read_mem(self, filename):
        """Read data from MEM file and return Stims and Amps."""
        lines = self._read_file_lines(filename)

        ms_indices = [i for i, line in enumerate(lines) if "MS." in line]
        
        if not ms_indices:
            return [], []
        
        scanpts_index=ms_indices[0]-2

        scanpts=lines[scanpts_index].split()
        prescan_indices=[int(scanpts[1][:-1]), int(scanpts[2][:-1])]
        postscan_indices=[int(scanpts[3][:-1]), int(scanpts[4])]
        

        stims, amps = [], []
        for i in ms_indices:
            parts = lines[i].split()
            stims.append(float(parts[1]))
            amps.append(float(parts[2]))

        return np.array(stims), np.array(amps), prescan_indices, postscan_indices

    def _find_name(self, filename):
        lines = self._read_file_lines(filename)
        MSindices = [i for i, line in enumerate(lines) if "Name:" in line]
        if len(MSindices)==0:
            return None
        MSlines=[lines[i] for i in MSindices]
        for line in MSlines:
            index1=line.find("\t")
            NameString=line[index1+1:-1]
        if NameString[0]==" ":
            NameString=NameString[1:]
        return NameString

    def _read_file_lines(self, filename):
        """Read the lines of a file and return them as a list."""
        with open(filename, 'rt', encoding='utf-8', errors='replace') as file:
            return file.readlines()
        
    def get_mem_files_from_mef(mef_filepath):
        """
        Reads a .MEF file and returns a list of base filenames.

        .MEF files are expected to be simple text files with one .MEM
        file basename per line.

        Args:
            mef_filepath (str): The full path to the .MEF file.

        Returns:
            list: A list of strings, where each string is a filename
                (e.g., ['ME2C30612A_A1', 'ME2C30612A_A2', ...]).
                Returns an empty list if the file is not found.
        """
        try:
            with open(mef_filepath, 'rt') as file:
                # Read all lines, strip leading/trailing whitespace,
                # and filter out any empty lines that may exist.
                lines = [line.strip() for line in file.read().splitlines()]
                return [line for line in lines if line]
        except FileNotFoundError:
            print(f"Warning: MEF file not found at {mef_filepath}")
            return []