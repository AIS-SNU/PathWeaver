import os
import numpy as np
import argparse

#################################### VECS TO BIN ####################################

def bvecs_to_fbin(bvecs_file, fbin_file):
    """
    Convert a .bvecs file to .fbin format.
    
    Args:
        bvecs_file (str): Path to the input .bvecs file.
        fbin_file (str): Path to the output .fbin file.
    """
    data = []
    
    with open(bvecs_file, 'rb') as f:
        while True:
            # Read the vector length (4 bytes)
            length_bytes = f.read(4)
            if not length_bytes:
                break  # End of file
            
            vector_length = np.frombuffer(length_bytes, dtype=np.int32)[0]
            
            # Read the vector data
            vector_data = np.frombuffer(f.read(vector_length), dtype=np.uint8)
            data.append(vector_data.astype(np.float32))  # Convert to float32
    
    # Combine all vectors into a single NumPy array
    array_data = np.vstack(data)
    
    # Write to .fbin file
    with open(fbin_file, 'wb') as f:
        # Write the shape (vector_count, vector_length) as int32
        shape = np.array(array_data.shape, dtype=np.int32)
        shape.tofile(f)
        
        # Write the vector data as float32
        array_data.tofile(f)
    
    print(f"Converted {bvecs_file} to {fbin_file}")

def fvecs_to_fbin(fvecs_file, fbin_file):
    """
    Convert a .fvecs file to .fbin format.
    
    Args:
        fvecs_file (str): Path to the input .fvecs file.
        fbin_file (str): Path to the output .fbin file.
    """
    data = []

    with open(fvecs_file, 'rb') as f:
        while True:
            # Read the vector length (4 bytes)
            length_bytes = f.read(4)
            if not length_bytes:
                break  # End of file
            
            vector_length = np.frombuffer(length_bytes, dtype=np.int32)[0]
            
            # Read the vector data
            vector_data = np.frombuffer(f.read(4 * vector_length), dtype=np.float32)
            data.append(vector_data)
    
    # Combine all vectors into a single NumPy array
    array_data = np.vstack(data)
    
    # Write to .fbin file
    with open(fbin_file, 'wb') as f:
        # Write the shape (vector_count, vector_length) as int32
        shape = np.array(array_data.shape, dtype=np.int32)
        shape.tofile(f)
        
        # Write the vector data as float32
        array_data.tofile(f)
    
    print(f"Converted {fvecs_file} to {fbin_file}")

def ivecs_to_ibin(ivecs_file, ibin_file):
    """
    Convert an .ivecs file to .ibin format.
    
    Args:
        ivecs_file (str): Path to the input .ivecs file.
        ibin_file (str): Path to the output .ibin file.
    """
    data = []

    with open(ivecs_file, 'rb') as f:
        while True:
            # Read the vector length (4 bytes)
            length_bytes = f.read(4)
            if not length_bytes:
                break  # End of file
            
            vector_length = np.frombuffer(length_bytes, dtype=np.int32)[0]
            
            # Read the vector data
            vector_data = np.frombuffer(f.read(4 * vector_length), dtype=np.int32)
            data.append(vector_data)
    
    # Combine all vectors into a single NumPy array
    array_data = np.vstack(data)
    
    # Write to .ibin file
    with open(ibin_file, 'wb') as f:
        # Write the shape (vector_count, vector_length) as int32
        shape = np.array(array_data.shape, dtype=np.int32)
        shape.tofile(f)
        
        # Write the vector data as int32
        array_data.tofile(f)
    
    print(f"Converted {ivecs_file} to {ibin_file}")

#################################### BIN TO VECS ####################################

def fbin_to_fvecs(fbin_file, fvecs_file):
    """
    Convert a .fbin file back to .fvecs format.
    
    The .fbin file format:
        - Header: two int32 numbers: [num_vectors, vector_length]
        - Data: all vector data as float32 in row-major order.
    
    The .fvecs file format:
        For each vector:
            - Write vector length as int32 (4 bytes)
            - Write vector data as float32 (4 bytes each)
    
    Args:
        fbin_file (str): Path to the input .fbin file.
        fvecs_file (str): Path to the output .fvecs file.
    """
    with open(fbin_file, 'rb') as f:
        # Read header: two int32 numbers: (num_vectors, vector_length)
        header = np.fromfile(f, dtype=np.int32, count=2)
        if header.size < 2:
            raise ValueError("Invalid fbin file: header too short.")
        num_vectors, vector_length = header
        # Read all vector data as float32
        data = np.fromfile(f, dtype=np.float32)
    
    # Reshape data to (num_vectors, vector_length)
    data = data.reshape((num_vectors, vector_length))
    
    # Write to .fvecs file: for each vector, write length then data
    with open(fvecs_file, 'wb') as f:
        for i in range(num_vectors):
            # Write vector length as int32
            np.array(vector_length, dtype=np.int32).tofile(f)
            # Write vector data as float32
            data[i].tofile(f)
    
    print(f"Converted {fbin_file} to {fvecs_file}")


def ibin_to_ivecs(ibin_file, ivecs_file):
    """
    Convert an .ibin file back to .ivecs format.
    
    The .ibin file format:
        - Header: two int32 numbers: [num_vectors, vector_length]
        - Data: all vector data as int32 in row-major order.
    
    The .ivecs file format:
        For each vector:
            - Write vector length as int32 (4 bytes)
            - Write vector data as int32 (4 bytes each)
    
    Args:
        ibin_file (str): Path to the input .ibin file.
        ivecs_file (str): Path to the output .ivecs file.
    """
    with open(ibin_file, 'rb') as f:
        # Read header: two int32 numbers: (num_vectors, vector_length)
        header = np.fromfile(f, dtype=np.int32, count=2)
        if header.size < 2:
            raise ValueError("Invalid ibin file: header too short.")
        num_vectors, vector_length = header
        # Read all vector data as int32
        data = np.fromfile(f, dtype=np.int32)
    
    # Reshape data to (num_vectors, vector_length)
    data = data.reshape((num_vectors, vector_length))
    
    # Write to .ivecs file: for each vector, write length then data
    with open(ivecs_file, 'wb') as f:
        for i in range(num_vectors):
            # Write vector length as int32
            np.array(vector_length, dtype=np.int32).tofile(f)
            # Write vector data as int32
            data[i].tofile(f)
    
    print(f"Converted {ibin_file} to {ivecs_file}")


################################### CHECK FORMAT ####################################

def read_fbin_ibin_format(file_path, header_size = 8):
    
    with open(file_path, 'rb') as f:

        header = f.read(header_size)
        part1 = int.from_bytes(header[0:4], byteorder='little', signed=True)
        part2 = int.from_bytes(header[4:8], byteorder='little', signed=True)

        print(f"Header(raw): {header}")
        print(f"Header[0:4] as int32: {part1}, Header[4:8] as int32: {part2}")
        

def main():

    parser = argparse.ArgumentParser(description='Convert .fbin and .ibin files to .fvecs and .ivecs formats.')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--base_path', type=str, required=True, help='Base path for the dataset')
    
    args = parser.parse_args()

    dataset_path = f"{args.base_path}/{args.dataset_name}/"

    print(f"Processing dataset: {args.dataset_name}")

    for file_name in ["base.fbin", "groundtruth.distances.fbin", "groundtruth.neighbors.ibin", "query.fbin"]:

        file_format = file_name.split(".")[-1]

        if file_format == "fbin":
            fbin_file = dataset_path + file_name
            if not os.path.exists(fbin_file.replace(".fbin", ".fvecs")):
                read_fbin_ibin_format(fbin_file)
                fbin_to_fvecs(fbin_file, fbin_file.replace(".fbin", ".fvecs"))
        elif file_format == "ibin":
            ibin_file = dataset_path + file_name
            if not os.path.exists(ibin_file.replace(".ibin", ".ivecs")):
                read_fbin_ibin_format(ibin_file)
                ibin_to_ivecs(ibin_file, ibin_file.replace(".ibin", ".ivecs"))

if __name__ == "__main__":

    main()