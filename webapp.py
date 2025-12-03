import os
import sys

# Set matplotlib backend BEFORE any imports that might use it
# This is critical for web applications that run in threads
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for web/server use

from flask import Flask, render_template, request, jsonify, session
from threading import Thread
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Try to import - if this fails, show error immediately
try:
    from scripts.run_individual import run_individual
    from scripts.run_batch import run_batch
except ImportError as e:
    print(f"ERROR: Failed to import processing modules: {e}")
    print("\nMake sure you:")
    print("1. Have activated the virtual environment: source init.sh")
    print("2. Installed all dependencies: pip install -r requirements.txt")
    print("3. Are running from the project root directory")
    sys.exit(1)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production

# Store processing status
processing_status = {
    'running': False,
    'progress': '',
    'error': None,
    'completed': False
}

def str2bool(value):
    """Convert string to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', 't', 'yes', 'y', '1')
    return bool(value)

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/individual')
def individual_form():
    """Individual processing form."""
    return render_template('individual.html')

@app.route('/batch')
def batch_form():
    """Batch processing form."""
    return render_template('batch.html')

@app.route('/api/process/individual', methods=['POST'])
def process_individual():
    """API endpoint for individual processing."""
    global processing_status
    
    if processing_status['running']:
        return jsonify({'error': 'Another process is already running'}), 400
    
    try:
        data = request.json
        
        # Extract parameters
        data_folder_path = data.get('data_folder_path')
        results_folder = data.get('results_folder', '')
        threshold_min_max = int(data.get('threshold_min_max', 1))
        threshold_pupil = int(data.get('threshold_pupil', 2))
        plot_traces = str2bool(data.get('plot_traces', False))
        save_trace_plot = str2bool(data.get('save_trace_plot', True))
        clear_output = str2bool(data.get('clear_output', False))
        bsline_length = int(data.get('bsline_length', 10))
        event_length = int(data.get('event_length', 20))
        wake_up = str2bool(data.get('wake_up', False))
        interactive_plots = str2bool(data.get('interactive_plots', False))
        
        if not data_folder_path:
            return jsonify({'error': 'Data folder path is required'}), 400
        
        # Reset status
        processing_status = {
            'running': True,
            'progress': 'Starting individual processing...',
            'error': None,
            'completed': False
        }
        
        # Run in background thread
        def run_process():
            global processing_status
            try:
                processing_status['progress'] = f'Processing folder: {data_folder_path}'
                
                run_individual(
                    data_folder_path=data_folder_path,
                    results_folder=results_folder if results_folder else None,
                    threshold_to_exclude_from_min_max=threshold_min_max,
                    threshold_to_exclude_base_on_pupil=threshold_pupil,
                    plot_traces=plot_traces,
                    save_trace_plot=save_trace_plot,
                    clear_output=clear_output,
                    bsline_length=bsline_length,
                    event_length=event_length,
                    wakeup=wake_up,
                    interactive_plots=interactive_plots
                )
                
                processing_status['progress'] = 'Processing completed successfully!'
                processing_status['running'] = False
                processing_status['completed'] = True
                
            except Exception as e:
                processing_status['error'] = str(e)
                processing_status['running'] = False
                processing_status['progress'] = f'Error: {str(e)}'
        
        thread = Thread(target=run_process)
        thread.daemon = True
        thread.start()
        
        return jsonify({'message': 'Processing started', 'status': 'running'})
        
    except Exception as e:
        processing_status['running'] = False
        return jsonify({'error': str(e)}), 500

@app.route('/api/process/batch', methods=['POST'])
def process_batch():
    """API endpoint for batch processing."""
    global processing_status
    
    if processing_status['running']:
        return jsonify({'error': 'Another process is already running'}), 400
    
    try:
        data = request.json
        
        # Extract parameters
        root_folder = data.get('root_folder', '')
        folders_list = data.get('folders_list', '').strip()
        results_path = data.get('results_path', '')
        threshold_min_max = int(data.get('threshold_min_max', 1))
        threshold_pupil = int(data.get('threshold_pupil', 2))
        plot_traces = str2bool(data.get('plot_traces', True))
        save_trace_plot = str2bool(data.get('save_trace_plot', True))
        clear_output = str2bool(data.get('clear_output', False))
        bsline_length = int(data.get('bsline_length', 5))
        event_length = int(data.get('event_length', 15))
        wake_up = str2bool(data.get('wake_up', False))
        interactive_plots = str2bool(data.get('interactive_plots', False))
        
        if not results_path:
            return jsonify({'error': 'Results path is required'}), 400
        
        if not root_folder and not folders_list:
            return jsonify({'error': 'Either root folder or folders list is required'}), 400
        
        # Parse folders list if provided
        list_of_folders = None
        if folders_list:
            list_of_folders = [f.strip() for f in folders_list.split(',') if f.strip()]
        
        # Reset status
        processing_status = {
            'running': True,
            'progress': 'Starting batch processing...',
            'error': None,
            'completed': False
        }
        
        # Run in background thread
        def run_process():
            global processing_status
            try:
                if root_folder:
                    processing_status['progress'] = f'Searching for folders in: {root_folder}'
                else:
                    processing_status['progress'] = f'Processing {len(list_of_folders)} folders...'
                
                run_batch(
                    root_folder=root_folder if root_folder else None,
                    list_of_folders=list_of_folders,
                    results_path=results_path,
                    threshold_to_exclude_from_min_max=threshold_min_max,
                    threshold_to_exclude_base_on_pupil=threshold_pupil,
                    plot_traces=plot_traces,
                    save_trace_plot=save_trace_plot,
                    clear_output=clear_output,
                    bsline_length=bsline_length,
                    event_length=event_length,
                    wakeup=wake_up,
                    interactive_plots=interactive_plots
                )
                
                processing_status['progress'] = 'Batch processing completed successfully!'
                processing_status['running'] = False
                processing_status['completed'] = True
                
            except Exception as e:
                processing_status['error'] = str(e)
                processing_status['running'] = False
                processing_status['progress'] = f'Error: {str(e)}'
        
        thread = Thread(target=run_process)
        thread.daemon = True
        thread.start()
        
        return jsonify({'message': 'Processing started', 'status': 'running'})
        
    except Exception as e:
        processing_status['running'] = False
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get processing status."""
    return jsonify(processing_status)

@app.route('/api/reset', methods=['POST'])
def reset_status():
    """Reset processing status."""
    global processing_status
    processing_status = {
        'running': False,
        'progress': '',
        'error': None,
        'completed': False
    }
    return jsonify({'message': 'Status reset'})

@app.route('/api/browse', methods=['GET'])
def browse_directory():
    """Browse directory contents."""
    try:
        path = request.args.get('path', '.')
        
        # Security: Ensure path is within project directory or absolute path
        project_root = os.path.abspath(os.path.dirname(__file__))
        
        if os.path.isabs(path):
            # For absolute paths, ensure it exists
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path):
                return jsonify({'error': 'Path does not exist'}), 404
        else:
            # For relative paths, resolve relative to project root
            abs_path = os.path.abspath(os.path.join(project_root, path))
            if not os.path.exists(abs_path):
                return jsonify({'error': 'Path does not exist'}), 404
        
        if not os.path.isdir(abs_path):
            return jsonify({'error': 'Path is not a directory'}), 400
        
        # Get parent directory
        parent_path = os.path.dirname(abs_path)
        if parent_path == abs_path:
            parent_path = None  # At root, no parent
        
        # List directory contents
        items = []
        try:
            for item in sorted(os.listdir(abs_path)):
                item_path = os.path.join(abs_path, item)
                # Skip hidden files/folders (starting with .) except common ones
                if item.startswith('.') and item not in ['.git', '.gitignore']:
                    continue
                
                try:
                    is_dir = os.path.isdir(item_path)
                    items.append({
                        'name': item,
                        'path': item_path,
                        'is_directory': is_dir,
                        'relative_path': os.path.relpath(item_path, project_root)
                    })
                except (OSError, PermissionError):
                    continue  # Skip items we can't access
        except PermissionError:
            return jsonify({'error': 'Permission denied'}), 403
        
        # Calculate relative path safely
        try:
            if abs_path.startswith(project_root):
                relative_path = os.path.relpath(abs_path, project_root)
            else:
                relative_path = abs_path
        except (ValueError, AttributeError):
            relative_path = abs_path
        
        return jsonify({
            'current_path': abs_path,
            'relative_path': relative_path,
            'parent_path': parent_path,
            'items': items
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/resolve-folder-path', methods=['POST'])
def resolve_folder_path():
    """Resolve folder path from uploaded files."""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if len(files) == 0:
            return jsonify({'error': 'No files provided'}), 400
        
        # Get the first file's path
        first_file = files[0]
        
        # Try to get the actual file path from the uploaded file
        # Note: Browsers don't expose full paths, but we can use the filename and webkitRelativePath
        # For now, we'll extract what we can and let the user verify
        
        # Get webkitRelativePath if available (from the original file selection)
        # Since files are uploaded, we need to work with what we have
        
        # For local filesystem access, we can try to find the file
        # But this is limited by browser security
        
        # Extract folder structure from filename patterns
        # The best we can do is return the folder name/structure
        # Actual path resolution would require the files to already be on the server
        
        # Return a message that user should verify the path
        # In practice, webkitRelativePath gives us the folder structure relative to selected folder
        folder_name = os.path.dirname(first_file.filename) if '/' in first_file.filename else ''
        
        return jsonify({
            'resolved_path': folder_name,
            'note': 'Please verify the folder path is correct. Browser security limits full path access.'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/check-csv', methods=['GET'])
def check_csv_files():
    """Check if a directory contains CSV files."""
    try:
        path = request.args.get('path', '')
        if not path:
            return jsonify({'has_csv': False, 'csv_files': []})
        
        project_root = os.path.abspath(os.path.dirname(__file__))
        
        # Handle both relative and absolute paths
        if os.path.isabs(path):
            abs_path = os.path.abspath(path)
        else:
            abs_path = os.path.abspath(os.path.join(project_root, path))
        
        if not os.path.exists(abs_path) or not os.path.isdir(abs_path):
            return jsonify({'has_csv': False, 'csv_files': []})
        
        csv_files = []
        try:
            for item in os.listdir(abs_path):
                if item.lower().endswith('.csv'):
                    csv_files.append(item)
        except (OSError, PermissionError):
            pass
        
        # Check for common data file patterns
        has_data_files = False
        patterns = ['calcium', 'pupil', 'arteriole', 'whisker']
        all_files = [f.lower() for f in os.listdir(abs_path) if os.path.isfile(os.path.join(abs_path, f))]
        for pattern in patterns:
            if any(pattern in f for f in all_files):
                has_data_files = True
                break
        
        return jsonify({
            'has_csv': len(csv_files) > 0 or has_data_files,
            'csv_files': csv_files[:10],  # Limit to first 10
            'csv_count': len(csv_files)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import socket
    
    def find_free_port(start_port=5000, max_attempts=10):
        """Find a free port starting from start_port."""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return None
    
    # Get port from environment variable or find a free one
    port = int(os.environ.get('FLASK_PORT', 0))
    if port == 0:
        port = find_free_port(5000)
        if port is None:
            print("Error: Could not find a free port. Please free up ports 5000-5010")
            sys.exit(1)
    
    print("\n" + "="*50)
    print("Pupil-Based Waking Up Event Detection - Web UI")
    print("="*50)
    print(f"\nStarting web server on port {port}...")
    print(f"Open your browser and go to: http://localhost:{port}")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        # Add use_reloader=False to avoid double startup
        app.run(debug=True, host='127.0.0.1', port=port, threaded=True, use_reloader=False)
        print(f"\n✓ Server started successfully on http://127.0.0.1:{port}")
    except OSError as e:
        if 'Address already in use' in str(e):
            # Try next port
            port = find_free_port(port + 1)
            if port:
                print(f"\nPort {port - 1} was in use. Using port {port} instead.")
                print(f"Open your browser and go to: http://127.0.0.1:{port}\n")
                app.run(debug=True, host='127.0.0.1', port=port, threaded=True, use_reloader=False)
            else:
                print("Error: Could not find a free port. Please free up some ports.")
                sys.exit(1)
        else:
            raise
    except Exception as e:
        print(f"\nERROR: Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)