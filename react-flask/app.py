import os
import subprocess
from flask_cors import CORS
from flask import Flask, jsonify, send_from_directory, request

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Ensure the outputs folder exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def get_next_filename(folder, prefix, extension):
    """
    Generate the next filename in the form of 'Attendance[i].ext',
    where i is the next available number and .ext is the extension.
    """
    existing_files = os.listdir(folder)
    attendance_files = [f for f in existing_files if f.startswith(prefix) and f.endswith(f'.{extension}')]

    numbers = []
    for f in attendance_files:
        try:
            # Extract the number part of the filename, after the prefix and before the extension
            number = int(f.replace(prefix, '').split('.')[0])
            numbers.append(number)
        except ValueError:
            # Ignore files that don't have a valid number after the prefix
            continue
    
    if numbers:
        next_number = max(numbers) + 1
    else:
        next_number = 1  # Start from 1 if no valid files exist

    return f'{prefix}{next_number}.{extension}'


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Generate the next available Attendance[i] filename for the upload
        new_filename = get_next_filename(UPLOAD_FOLDER, 'Attendance', os.path.splitext(file.filename)[1][1:])  # Retain the original extension
        file_path = os.path.join(UPLOAD_FOLDER, new_filename)
        
        # Save the file in the uploads folder
        file.save(file_path)

        # Generate the next available output filename
        output_filename = get_next_filename(OUTPUT_FOLDER, 'Attendance', 'csv')
        
        # Run Core.py and pass the output filename
        try:
            print(f"Saving file to {file_path}")
            subprocess.run(['python', 'Core.py', output_filename], check=True)  # Pass output file name as an argument
            print("Core.py executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Core.py execution failed: {e}")
            return jsonify({"error": f"Core.py execution failed: {e}"}), 500

    return jsonify({"message": "File uploaded and processing started", "new_filename": new_filename, "output_filename": output_filename}), 200

@app.route('/api/latest-processed-file', methods=['GET'])
def get_latest_processed_file():
    output_file = get_latest_output_file()
    if output_file:
        return jsonify({"output_file": output_file}), 200
    else:
        return jsonify({"output_file": None}), 404  # Not found if no file exists

def get_latest_output_file():
    existing_files = os.listdir(OUTPUT_FOLDER)
    attendance_files = [f for f in existing_files if f.startswith('Attendance') and f.endswith('.csv')]

    if attendance_files:
        numbers = []
        for f in attendance_files:
            # Extract the number part from the filename
            number_str = f.replace('Attendance', '').replace('.csv', '')
            if number_str.isdigit():  # Check if it's a valid number
                numbers.append(int(number_str))

        if numbers:  # Ensure there are valid numbers before finding the max
            max_index = max(numbers)
            latest_file = f'Attendance{max_index}.csv'
            return latest_file  # Return the latest processed file
    
    return None  # Return None if no valid attendance files found


@app.route('/api/output/<filename>', methods=['GET'])
def download_output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
