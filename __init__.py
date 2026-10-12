# Version of the Namespace package "dummy"
__version__ = "2021.01.a1"

# contains also script to auto-compile .proto files into python classes. adapted from
# https://stackoverflow.com/questions/23585450/automate-compilation-of-protobuf-specs-into-python-classes-in-setup-py
import os, sys, subprocess
from pathlib import Path
from distutils.spawn import find_executable
import tempfile

# find protoc installation
if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
  protoc = os.environ['PROTOC']
else:
  protoc = find_executable('protoc')

# search all proto files in this package
this_dir = os.path.dirname(os.path.realpath(__file__)) # this dir
working_dir = os.path.join(this_dir, 'identification')
proto_files_g = list(Path(working_dir).rglob("*.[pP][rR][oO][tT][oO]"))
python_out = os.path.join(working_dir, '_proto_gen')

# find global protobuf identification file
for path in proto_files_g:
    path_dir = os.path.dirname(path)

    if path.parts[-3:] == ('feature', 'identification', 'identification.proto'):
        id_path_dir = path_dir

id_path = os.path.join(id_path_dir, "identification.proto")
if not os.path.exists(id_path):
    print("Could not find identification.proto main file.")

num_compied_files = 0
# create pb2 and put them in the correct directory

for path in proto_files_g:
    path_dir = os.path.dirname(path)
    output_file = os.path.join(python_out, os.path.basename(path).split('.')[0] + '_pb2.py')

    if path_dir == id_path_dir:
        continue
        
    if '.ipynb_checkpoints' in str(path): # skip ipynb
        continue

    # generate if one of source files newer or output file non existent
    if not os.path.exists(output_file) or (os.path.getmtime(path) > os.path.getmtime(output_file)) or (os.path.getmtime(id_path) > os.path.getmtime(output_file)):
        if protoc is None:
            print("Protoc recompilation needed, but no proto-compiler found. Install it, e.g. protobuf-compiler via apt.")
            sys.exit(1)

        # merge path_dir and id_path_dir files (Properties by method and base file)
        # wish there was an easier solution
        # create temporal file to merge them, compile, and delete at end
        with tempfile.NamedTemporaryFile(delete=False) as tmp:

            files = [path, id_path]
            for fname in files:
                with open(fname, 'rb') as infile:
                    tmp.write(infile.read())
            tmp.flush()

            protoc_command = [protoc, '-I=' + os.path.dirname(tmp.name),  '--python_out=' + python_out, tmp.name]
            print("Execute protoc for: " + str(path))

        # need to call protoc outside (for windows compatibility), otherwise protoc cant access file.
        if subprocess.call(protoc_command) != 0:
            sys.exit(1)
        num_compied_files += 1

        os.replace(
                os.path.join(python_out, os.path.basename(tmp.name)) + "_pb2.py",
                os.path.join(python_out + "/" + os.path.basename(path)[:-6]) + "_pb2.py"
                 )
        # delete temp now
        os.remove(tmp.name)

if num_compied_files:
    print("Compiled " + str(num_compied_files) + " proto files.")
