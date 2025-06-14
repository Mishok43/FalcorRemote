#!/usr/bin/env python3

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
import io
import numpy as np
from PIL import Image
import threading
import time
import json
import os
import uuid
from datetime import datetime
import traceback

from basic_rasterizer import BasicRasterizer
import threading

app = Flask(__name__)
CORS(app)  # Enable CORS for web access

# Global renderer and synchronization
global_renderer = None
renderer_lock = threading.Lock()  # Ensure sequential access
renderer_initialized = False


# Global state
active_sessions = {}  # session_id -> session info
render_jobs = {}      # job_id -> job status

def create_session_id():
    """Generate unique session ID"""
    return str(uuid.uuid4())[:8]

def create_job_id():
    """Generate unique job ID"""
    return str(uuid.uuid4())[:12]

def initialize_global_renderer(render_width=512, render_height=512, use_nf_sampler=True,
                              train_batch_size=50, enable_3d_viz=False, verbose=True):
    """Initialize the global renderer with Cornell Box scene (thread-safe)"""
    global global_renderer, renderer_initialized

    with renderer_lock:
        if not renderer_initialized:
            print("Initializing global renderer with Cornell Box scene...")
            try:
                global_renderer = BasicRasterizer(
                    render_width=render_width,
                    render_height=render_height,
                    verbose=verbose,
                    use_nf_sampler=use_nf_sampler,
                    train_batch_size=train_batch_size,
                    enable_3d_viz=enable_3d_viz
                )

                print("Setting up Falcor...")
                global_renderer.setup_falcor()

                print("Loading Cornell Box scene...")
                # Load Cornell Box scene at startup
                success = global_renderer.set_current_scene("cornell_box")
                if not success:
                    raise Exception("Failed to load Cornell Box scene")

                print("Setting up render graph...")
                global_renderer.setup_render_graph()

                print("Performing initial render sample to keep window alive...")
                renderer_initialized = True
                print("Global renderer initialized successfully with Cornell Box!")
                print(f"Scene bounds: {global_renderer.scene_bounds}")
                #time.sleep(100)
                return True


            except Exception as e:
                print(f"Failed to initialize global renderer: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("Global renderer already initialized")
            return True

initialize_global_renderer(verbose=False)


@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    return jsonify({
        "message": "Adaptive Camera Sampling API",
        "version": "2.0",
        "endpoints": {
            "/scenes": "GET - List available scenes",
            "/setup_renderer": "POST - Setup renderer instance",
            "/set_scene": "POST - Set current scene for a renderer",
            "/sample_render": "POST - Sample render positions with albedo",
            "/train": "POST - Train NFSampler for X steps",
            "/render": "POST - Request render samples (legacy)",
            "/status/<job_id>": "GET - Check render job status",
            "/sessions": "GET - List active sessions"
        },
        "new_features": {
            "scene_management": "Get available scenes and set current scene",
            "sample_render": "Sample X positions with albedo renders and camera info",
            "custom_training": "Train with custom parameters and step count",
            "nn_sampling": "Toggle neural network sampling on/off"
        },
        "example_workflow": [
            "1. GET /scenes - See available scenes",
            "2. POST /setup_renderer - Setup renderer and get session_id",
            "3. POST /set_scene - Set current scene",
            "4. POST /sample_render - Sample positions with albedo",
            "5. POST /train - Train NFSampler with custom parameters"
        ]
    })

@app.route('/scenes', methods=['GET'])
def list_scenes():
    """List available scenes"""
    try:
        # Initialize global renderer if needed
        if not renderer_initialized:
            success = initialize_global_renderer(verbose=False)
            time.sleep(100)
            if not success:
                return jsonify({'error': 'Failed to initialize renderer'}), 500

        with renderer_lock:
            available_scenes = global_renderer.get_available_scenes()

        return jsonify({
            "available_scenes": available_scenes,
            "total_scenes": len(available_scenes),
            "existing_scenes": {k: v for k, v in available_scenes.items() if v["exists"]}
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get scene list: {str(e)}'}), 500

@app.route('/setup_renderer', methods=['POST'])
def setup_renderer():
    """Setup a renderer session (uses global renderer)"""
    try:
        data = request.json or {}
        render_width = data.get('render_width', 512)
        render_height = data.get('render_height', 512)
        use_nf_sampler = data.get('use_nf_sampler', True)
        train_batch_size = data.get('train_batch_size', 50)
        enable_3d_viz = data.get('enable_3d_viz', False)
        verbose = data.get('verbose', True)

        # Create session ID
        session_id = create_session_id()

        # Initialize global renderer if needed
        success = initialize_global_renderer(
            render_width=render_width,
            render_height=render_height,
            use_nf_sampler=use_nf_sampler,
            train_batch_size=train_batch_size,
            enable_3d_viz=enable_3d_viz,
            verbose=verbose
        )

        if not success:
            return jsonify({
                'error': 'Failed to initialize global renderer',
                'session_id': session_id
            }), 500

        # Create session info (Cornell Box is pre-loaded)
        active_sessions[session_id] = {
            'setup_time': datetime.now().isoformat(),
            'status': 'ready',
            'current_scene': 'cornell_box',  # Pre-loaded scene
            'total_samples_rendered': 0,
            'total_training_steps': 0,
            'render_config': {
                'width': render_width,
                'height': render_height,
                'use_nf_sampler': use_nf_sampler,
                'train_batch_size': train_batch_size,
                'enable_3d_viz': enable_3d_viz,
                'verbose': verbose
            }
        }

        print(f"Session {session_id} created successfully (using global renderer)")

        return jsonify({
            'session_id': session_id,
            'status': 'ready',
            'current_scene': 'cornell_box',
            'message': 'Session created successfully. Cornell Box scene is pre-loaded and ready.',
            'scene_bounds': {
                'min': global_renderer.scene_bounds[0].tolist(),
                'max': global_renderer.scene_bounds[1].tolist()
            },
            'render_config': {
                'width': render_width,
                'height': render_height,
                'use_nf_sampler': use_nf_sampler,
                'train_batch_size': train_batch_size,
                'enable_3d_viz': enable_3d_viz,
                'verbose': verbose
            }
        })

    except Exception as e:
        return jsonify({'error': f'Setup failed: {str(e)}'}), 500

@app.route('/set_scene', methods=['POST'])
def set_scene():
    """Scene is fixed to Cornell Box - this endpoint is disabled"""
    return jsonify({
        'message': 'Scene is fixed to Cornell Box for stability.',
        'current_scene': 'cornell_box',
        'note': 'Scene switching is disabled to prevent window closing issues.'
    })

@app.route('/sample_render', methods=['POST'])
def sample_render():
    """Sample render positions with albedo returns"""
    try:
        data = request.json
        session_id = data.get('session_id')
        num_samples = data.get('num_samples', 10)
        nn_sampling = data.get('nn_sampling', False)
        return_albedo = data.get('return_albedo', True)

        if not session_id:
            return jsonify({'error': 'session_id required'}), 400

        if session_id not in active_sessions:
            return jsonify({'error': f'Session {session_id} not found'}), 404

        session_data = active_sessions[session_id]

        if session_data['status'] != 'ready':
            return jsonify({
                'error': f'Session not ready. Status: {session_data["status"]}',
                'session_id': session_id
            }), 400

        # Cornell Box is pre-loaded, no need to check scene

        # Check if global renderer is initialized
        if not renderer_initialized or global_renderer is None:
            return jsonify({
                'error': 'Global renderer not initialized',
                'session_id': session_id
            }), 500

        # Create render job
        job_id = create_job_id()

        def sample_render_async():
            try:
                render_jobs[job_id] = {
                    'status': 'running',
                    'session_id': session_id,
                    'num_samples': num_samples,
                    'nn_sampling': nn_sampling,
                    'return_albedo': return_albedo,
                    'start_time': datetime.now().isoformat(),
                    'progress': 0
                }

                # Use global renderer with lock for sequential access
                with renderer_lock:
                    results = global_renderer.sample_render_positions(
                        num_samples=num_samples,
                        nn_sampling=nn_sampling,
                        return_albedo=return_albedo
                    )

                # Update session stats
                session_data['total_samples_rendered'] += num_samples
                session_data['last_render_time'] = datetime.now().isoformat()

                # Complete job
                render_jobs[job_id].update({
                    'status': 'completed',
                    'results': results,
                    'end_time': datetime.now().isoformat(),
                    'total_samples': len(results),
                    'progress': 100
                })

                print(f"Sample render job {job_id} completed with {len(results)} samples")

            except Exception as e:
                render_jobs[job_id] = {
                    'status': 'error',
                    'error': str(e),
                    'session_id': session_id,
                    'start_time': datetime.now().isoformat(),
                    'end_time': datetime.now().isoformat()
                }
                print(f"Sample render job {job_id} failed: {e}")
                traceback.print_exc()

        # Start render in background
        threading.Thread(target=sample_render_async, daemon=True).start()

        return jsonify({
            'job_id': job_id,
            'session_id': session_id,
            'status': 'started',
            'num_samples': num_samples,
            'nn_sampling': nn_sampling,
            'return_albedo': return_albedo,
            'message': f'Sample render job started. Check /status/{job_id} for progress.'
        })

    except Exception as e:
        return jsonify({'error': f'Sample render request failed: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train_nf_sampler():
    """Train NFSampler for X steps with custom parameters"""
    try:
        data = request.json
        session_id = data.get('session_id')
        num_steps = data.get('num_steps', 100)

        # Extract training parameters
        training_params = {}
        if 'learning_rate' in data:
            training_params['learning_rate'] = data['learning_rate']
        if 'epochs_per_fit' in data:
            training_params['epochs_per_fit'] = data['epochs_per_fit']
        if 'batch_size' in data:
            training_params['batch_size'] = data['batch_size']
        if 'hidden_units' in data:
            training_params['hidden_units'] = data['hidden_units']
        if 'hidden_layers' in data:
            training_params['hidden_layers'] = data['hidden_layers']

        if not session_id:
            return jsonify({'error': 'session_id required'}), 400

        if session_id not in active_sessions:
            return jsonify({'error': f'Session {session_id} not found'}), 404

        session_data = active_sessions[session_id]

        if session_data['status'] != 'ready':
            return jsonify({
                'error': f'Session not ready. Status: {session_data["status"]}',
                'session_id': session_id
            }), 400

        if session_data['current_scene'] is None:
            return jsonify({
                'error': 'No scene set. Call /set_scene first.',
                'session_id': session_id
            }), 400

        # Check if global renderer is initialized
        if not renderer_initialized or global_renderer is None:
            return jsonify({
                'error': 'Global renderer not initialized',
                'session_id': session_id
            }), 500

        # Create training job
        job_id = create_job_id()

        def train_async():
            try:
                render_jobs[job_id] = {
                    'status': 'running',
                    'session_id': session_id,
                    'num_steps': num_steps,
                    'training_params': training_params,
                    'start_time': datetime.now().isoformat(),
                    'progress': 0
                }

                # Use global renderer with lock for sequential access
                with renderer_lock:
                    training_result = global_renderer.train_nf_sampler_steps(
                        num_steps=num_steps,
                        **training_params
                    )

                # Update session stats
                session_data['total_training_steps'] += num_steps
                session_data['last_training_time'] = datetime.now().isoformat()

                # Complete job
                render_jobs[job_id].update({
                    'status': 'completed',
                    'training_result': training_result,
                    'end_time': datetime.now().isoformat(),
                    'progress': 100
                })

                print(f"Training job {job_id} completed with {num_steps} steps")

            except Exception as e:
                render_jobs[job_id] = {
                    'status': 'error',
                    'error': str(e),
                    'session_id': session_id,
                    'start_time': datetime.now().isoformat(),
                    'end_time': datetime.now().isoformat()
                }
                print(f"Training job {job_id} failed: {e}")
                traceback.print_exc()

        # Start training in background
        threading.Thread(target=train_async, daemon=True).start()

        return jsonify({
            'job_id': job_id,
            'session_id': session_id,
            'status': 'started',
            'num_steps': num_steps,
            'training_params': training_params,
            'message': f'Training job started. Check /status/{job_id} for progress.'
        })

    except Exception as e:
        return jsonify({'error': f'Training request failed: {str(e)}'}), 500

@app.route('/render', methods=['POST'])
def request_render():
    """Request render samples"""
    try:
        data = request.json
        session_id = data.get('session_id')
        num_samples = data.get('num_samples', 10)

        if not session_id:
            return jsonify({'error': 'session_id required'}), 400

        if session_id not in active_sessions:
            return jsonify({'error': f'Session {session_id} not found'}), 404

        session_data = active_sessions[session_id]

        if session_data['status'] == 'setting_up':
            return jsonify({
                'error': 'Scene still setting up. Please wait.',
                'session_id': session_id,
                'status': 'setting_up'
            }), 202

        if session_data['status'] == 'error':
            return jsonify({
                'error': f'Session has error: {session_data.get("error", "Unknown error")}',
                'session_id': session_id
            }), 500

        if session_data['status'] != 'ready':
            return jsonify({
                'error': f'Session not ready. Status: {session_data["status"]}',
                'session_id': session_id
            }), 400

        # Create render job
        job_id = create_job_id()

        def render_async():
            try:
                render_jobs[job_id] = {
                    'status': 'running',
                    'session_id': session_id,
                    'num_samples': num_samples,
                    'start_time': datetime.now().isoformat(),
                    'progress': 0
                }

                renderer = session_data['renderer']
                results = renderer.render_samples(num_samples)

                # Convert results to API format
                api_results = []
                for result in results:
                    camera_info = result['camera_info']
                    api_results.append({
                        'sample_id': result['sample_id'],
                        'camera_position': camera_info['position'],
                        'camera_direction': camera_info['direction'],
                        'camera_target': camera_info['target'],
                        'surface_area': result['surface_area'],
                        'render_time': result['render_time'],
                        'sampling_method': result['sampling_method'],
                        'surface_area_stats': result.get('surface_area_stats', {})
                    })

                # Update session stats
                session_data['total_samples_rendered'] += num_samples
                session_data['last_render_time'] = datetime.now().isoformat()

                # Complete job
                render_jobs[job_id].update({
                    'status': 'completed',
                    'results': api_results,
                    'end_time': datetime.now().isoformat(),
                    'total_samples': len(api_results),
                    'progress': 100
                })

                print(f"Render job {job_id} completed with {len(api_results)} samples")

            except Exception as e:
                render_jobs[job_id] = {
                    'status': 'error',
                    'error': str(e),
                    'session_id': session_id,
                    'start_time': datetime.now().isoformat(),
                    'end_time': datetime.now().isoformat()
                }
                print(f"Render job {job_id} failed: {e}")
                traceback.print_exc()

        # Start render in background
        threading.Thread(target=render_async, daemon=True).start()

        return jsonify({
            'job_id': job_id,
            'session_id': session_id,
            'status': 'started',
            'num_samples': num_samples,
            'message': f'Render job started. Check /status/{job_id} for progress.'
        })

    except Exception as e:
        return jsonify({'error': f'Render request failed: {str(e)}'}), 500

@app.route('/status/<identifier>', methods=['GET'])
def get_status(identifier):
    """Get status of session or job"""
    # Check if it's a job ID
    if identifier in render_jobs:
        job = render_jobs[identifier]
        return jsonify({
            'type': 'job',
            'job_id': identifier,
            **job
        })

    # Check if it's a session ID
    if identifier in active_sessions:
        session = active_sessions[identifier]
        return jsonify({
            'type': 'session',
            'session_id': identifier,
            **session
        })

    return jsonify({'error': f'Identifier {identifier} not found'}), 404

@app.route('/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions"""
    sessions = {}
    for session_id, data in active_sessions.items():
        sessions[session_id] = {
            'current_scene': data.get('current_scene'),
            'status': data.get('status'),
            'setup_time': data.get('setup_time'),
            'total_samples_rendered': data.get('total_samples_rendered', 0),
            'total_training_steps': data.get('total_training_steps', 0)
        }

    return jsonify({
        'active_sessions': sessions,
        'total_sessions': len(sessions)
    })

@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all render jobs"""
    return jsonify({
        'render_jobs': render_jobs,
        'total_jobs': len(render_jobs)
    })

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Cleanup old sessions and jobs"""
    try:
        data = request.json or {}
        session_id = data.get('session_id')

        if session_id:
            # Cleanup specific session
            if session_id in active_sessions:
                del active_sessions[session_id]
                return jsonify({'message': f'Session {session_id} cleaned up'})
            else:
                return jsonify({'error': f'Session {session_id} not found'}), 404
        else:
            # Cleanup all
            active_sessions.clear()
            render_jobs.clear()
            return jsonify({'message': 'All sessions and jobs cleaned up'})

    except Exception as e:
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Adaptive Camera Sampling API Server...")
    print("Available endpoints:")
    print("  GET  /           - API documentation")
    print("  GET  /scenes     - List available scenes")
    print("  POST /setup_renderer - Setup renderer and get session_id")
    print("  POST /set_scene - Set current scene")
    print("  POST /sample_render - Sample positions with albedo")
    print("  POST /train - Train NFSampler with custom parameters")
    print("  POST /render     - Request render samples")
    print("  GET  /status/<id> - Check status")
    print("  GET  /sessions   - List active sessions")
    print("  GET  /jobs       - List render jobs")
    print("  POST /cleanup    - Cleanup sessions/jobs")
    print()
    print("Starting server on http://0.0.0.0:5000")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
