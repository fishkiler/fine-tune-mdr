Implement the following plan:                                                                                                                                                                                              
                                                                                                                                                                                                                         
  # Plan: Dashboard Training Controls + Logs + CPU Temp                                                                                                                                                                      
                                                                                                                                                                                                                         
  ## Context                                                                                                                                                                                                                 
                                                                                                                                                                                                                         
  The user needs to start/stop training on demand via the dashboard UI (not CLI), see live training logs, and monitor CPU temperature. The dashboard is a FastAPI + SSE app with vanilla JS frontend at port 28000. It   
  already has a subprocess management pattern (dataset refresh with start/cancel buttons + live log streaming) that we can reuse for training process management.                                                        
                                                                                                                                                                                                                         
  Additionally, the SIGINT handler needs fixing for clean stop/resume, and auto-resume should be the default.                                                                                                            
                                                                                                                                                                                                                         
  ## Changes                                                                                                                                                                                                             
                                                                                                                                                                                                                         
  ### 1. Training process management — `server.py`                                                                                                                                                                       
                                                                                                                                                                                                                         
  Add endpoints following the existing refresh subprocess pattern:                                                                                                                                                       
                                                                                                                                                                                                                         
  - `POST /train/start` — Spawns `bash train.sh` as async subprocess, captures stdout/stderr                                                                                                                             
  - `POST /train/stop` — Sends SIGINT to training process (cooperative shutdown)                                                                                                                                         
  - `GET /train/logs` — SSE endpoint streaming subprocess output (same as `/refresh/logs`)                                                                                                                               
  - `GET /train/state` — Returns process status (idle/running/stopping/complete/error)                                                                                                                                   
                                                                                                                                                                                                                         
  Server tracks `train_process_state` dict (same structure as `refresh_state`):                                                                                                                                          
  ```python                                                                                                                                                                                                              
  train_process_state = {                                                                                                                                                                                                
      "status": "idle",       # idle|running|stopping|complete|error                                                                                                                                                     
      "process": None,        # asyncio.subprocess.Process                                                                                                                                                               
      "logs": deque(maxlen=5000),                                                                                                                                                                                        
      "started_at": None,                                                                                                                                                                                                
      "stopped_at": None,                                                                                                                                                                                                
  }                                                                                                                                                                                                                      
  ```                                                                                                                                                                                                                    
                                                                                                                                                                                                                         
  The existing `train_state` (metrics from callback) stays unchanged — it receives structured metrics via `POST /log`. The new `train_process_state` manages the OS process and raw log output.                          
                                                                                                                                                                                                                         
  ### 2. CPU + GPU temperature — `callback.py`                                                                                                                                                                           
                                                                                                                                                                                                                         
  Add to `DashboardCallback._collect_gpu_metrics()`:                                                                                                                                                                     
                                                                                                                                                                                                                         
  ```python                                                                                                                                                                                                              
  # CPU temp — read from ACPI thermal zones                                                                                                                                                                              
  cpu_temps = []                                                                                                                                                                                                         
  for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):                                                                                                                                                  
      temp_file = zone / "temp"                                                                                                                                                                                          
      if temp_file.exists():                                                                                                                                                                                             
          cpu_temps.append(int(temp_file.read_text().strip()) / 1000)                                                                                                                                                    
  metrics["cpu_temp_c"] = max(cpu_temps) if cpu_temps else None                                                                                                                                                          
                                                                                                                                                                                                                         
  # GPU temp — via pynvml (already imported by torch.cuda)                                                                                                                                                               
  try:                                                                                                                                                                                                                   
      import pynvml                                                                                                                                                                                                      
      pynvml.nvmlInit()                                                                                                                                                                                                  
      handle = pynvml.nvmlDeviceGetHandleByIndex(0)                                                                                                                                                                      
      metrics["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)                                                                                                                       
  except Exception:                                                                                                                                                                                                      
      metrics["gpu_temp_c"] = None                                                                                                                                                                                       
  ```                                                                                                                                                                                                                    
                                                                                                                                                                                                                         
  ### 3. Fix SIGINT handler — `train_native.py`                                                                                                                                                                          
                                                                                                                                                                                                                         
  **Current** (broken): `trainer.save_model()` + `sys.exit(0)` — loses optimizer state.                                                                                                                                  
                                                                                                                                                                                                                         
  **New** (cooperative): Sets `trainer.control.should_training_stop = True` and `should_save = True`. The training loop finishes the current step and saves a proper numbered checkpoint. Second signal forces emergency 
   save + exit.                                                                                                                                                                                                          
                                                                                                                                                                                                                         
  ### 4. Auto-resume — `train_native.py`                                                                                                                                                                                 
                                                                                                                                                                                                                         
  Replace `--resume` (opt-in) with `--fresh` (opt-out). Default behavior: if a checkpoint exists, resume from it. The dashboard "Start Training" button runs `bash train.sh` which always auto-resumes.                  
                                                                                                                                                                                                                         
  ### 5. TimeLimitCallback — `dashboard/callback.py`                                                                                                                                                                     
                                                                                                                                                                                                                         
  Optional safety net (~25 lines). Sets `should_training_stop + should_save` in `on_step_end` after N hours. Configured via `time_limit_hours` in config.yaml. Default 0 (disabled).                                     
                                                                                                                                                                                                                         
  ### 6. Dashboard UI — `static/index.html`                                                                                                                                                                              
                                                                                                                                                                                                                         
  Add to the existing training dashboard page:                                                                                                                                                                           
                                                                                                                                                                                                                         
  **Training Control Panel** (new section, above metrics):                                                                                                                                                               
  - "Start Training" / "Stop Training" toggle button (same style as refresh buttons)                                                                                                                                     
  - Status badge: Idle → Starting → Training → Stopping → Complete                                                                                                                                                       
  - "Start Fresh" checkbox (passes `--fresh` flag)                                                                                                                                                                       
                                                                                                                                                                                                                         
  **Log Panel** (new section):                                                                                                                                                                                           
  - Scrollable log viewer (reuse refresh log panel CSS/JS pattern)                                                                                                                                                       
  - Auto-scroll with "pin to bottom" toggle                                                                                                                                                                              
  - SSE connection to `/train/logs`                                                                                                                                                                                      
  - Syntax coloring for INFO/WARNING/ERROR levels                                                                                                                                                                        
                                                                                                                                                                                                                         
  **Temperature Cards** (add to existing metrics row):                                                                                                                                                                   
  - CPU Temp card with color coding (green <60°C, yellow <75°C, red ≥75°C)                                                                                                                                               
  - GPU Temp card with same thresholds                                                                                                                                                                                   
  - Both updated via the existing `/metrics` SSE stream                                                                                                                                                                  
                                                                                                                                                                                                                         
  ### 7. Config — `config.yaml`                                                                                                                                                                                          
                                                                                                                                                                                                                         
  Add under `training:`:                                                                                                                                                                                                 
  ```yaml                                                                                                                                                                                                                
  time_limit_hours: 0  # 0 = no limit (manual stop). Set to 7.5 for nightly sessions.                                                                                                                                    
  ```                                                                                                                                                                                                                    
                                                                                                                                                                                                                         
  ## Files Modified                                                                                                                                                                                                      
                                                                                                                                                                                                                         
  | File | Change | Scope |                                                                                                                                                                                              
  |------|--------|-------|                                                                                                                                                                                              
  | `dashboard/server.py` | Add training process management endpoints (start/stop/logs/state) | ~80 lines |                                                                                                              
  | `dashboard/callback.py` | Add CPU/GPU temp to metrics, add `TimeLimitCallback` class | ~50 lines |                                                                                                                   
  | `dashboard/static/index.html` | Add control panel, log viewer, temp cards | ~200 lines |                                                                                                                             
  | `train_native.py` | Fix SIGINT handler, `--resume` → `--fresh`, wire TimeLimitCallback | ~30 lines |                                                                                                                 
  | `config.yaml` | Add `time_limit_hours: 0` | 1 line |                                                                                                                                                                 
                                                                                                                                                                                                                         
  ## Verification                                                                                                                                                                                                        
                                                                                                                                                                                                                         
  1. Start dashboard: `uvicorn dashboard.server:app --host 0.0.0.0 --port 28000`                                                                                                                                         
  2. Open `http://192.168.1.205:28000/` — confirm new panels visible                                                                                                                                                     
  3. Click "Start Training" — confirm training subprocess spawns, logs stream, metrics flow                                                                                                                              
  4. Click "Stop Training" — confirm cooperative shutdown, checkpoint saved                                                                                                                                              
  5. Click "Start Training" again — confirm auto-resume from checkpoint                                                                                                                                                  
  6. Verify CPU/GPU temp cards update with each metrics tick                                                                                                                                                             
  7. Kill training with Ctrl+C in terminal — confirm same clean checkpoint behavior                                                                                                                                      
                                                                                                                                                                                                                         
                                                                                                                                                                                                                         
  If you need specific details from before exiting plan mode (like exact code snippets, error messages, or content you generated), read the full transcript at:                                                          
  /home/jayoung/.claude/projects/-home-jayoung-Documents-dgx-code-bank-fine-tune-mdr/b2c817d9-439d-42ae-9258-78c900247dba.jsonl    



                                                                                                                                                                                          
● All changes implemented and verified. Here's a summary:                                                                                                                               
                                                                                                                                                                                        
  Changes Made                                                                                                                                                                          
                                                                                         
  File: dashboard/server.py                                                                                                                                                             
  What Changed: Added train_process_state dict, broadcast_train_log(), and 4 endpoints: POST /train/start, POST /train/stop, GET /train/logs (SSE), GET /train/state                    
  ────────────────────────────────────────                                                                                                                                              
  File: dashboard/callback.py                                                                                                                                                           
  What Changed: Added CPU temp (ACPI thermal zones) and GPU temp (pynvml) to _gpu_metrics(). Added TimeLimitCallback class (~25 lines)                                                  
  ────────────────────────────────────────                                                                                                                                              
  File: train_native.py                                                                                                                                                                 
  What Changed: Replaced broken SIGINT handler with cooperative shutdown (should_training_stop + should_save). Changed --resume to --fresh (auto-resume is now default). Wired          
    TimeLimitCallback                                                                                                                                                                   
  ────────────────────────────────────────
  File: config.yaml
  What Changed: Added time_limit_hours: 0
  ────────────────────────────────────────
  File: dashboard/static/index.html
  What Changed: Added Training Control Panel (start/stop buttons, status badge, fresh checkbox, log viewer with SSE streaming). Added CPU/GPU Temp cards with color coding. All wired
    with init-time state restore

  Key Behaviors

  - Start Training: Spawns bash train.sh as subprocess, streams stdout to log panel via SSE
  - Stop Training: Sends SIGINT for cooperative shutdown — trainer finishes current step, saves a proper numbered checkpoint with optimizer state
  - Auto-resume: Default behavior. If checkpoints exist, training resumes from the latest one. Use "Start Fresh" checkbox to override
  - Time limit: Disabled by default (time_limit_hours: 0). Set to 7.5 for nightly 8-hour sessions
  - Temperature: CPU (max across thermal zones) and GPU (via pynvml) updated with each metrics tick, with green/yellow/red color codingWhat is