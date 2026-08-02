#!/usr/bin/env python3
"""
Test executor for running RCCL tests with different configurations.
Builds mpirun commands and executes tests with proper environment setup.
"""

import subprocess
import time
import os
import shlex
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path


class SweepExecutor:
    """Executes RCCL tests with specified parameters."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str, verbose: bool = True):
        """Initialize the executor.
        
        Args:
            config: Configuration dictionary from sweep_config.yaml
            output_dir: Directory for output files
            verbose: Whether to print verbose output
        """
        self.config = config
        self.verbose = verbose
        
        # Set up output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def read_servers(self, servers_file: str) -> List[str]:
        """Read server IPs from servers.txt file.
        
        Args:
            servers_file: Path to servers file
            
        Returns:
            List of server IP addresses
        """
        servers = []
        with open(servers_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and lines starting with #
                if not line or line.startswith('#'):
                    continue
                # Remove inline comments (anything after #)
                if '#' in line:
                    line = line.split('#')[0].strip()
                # Take only the IP part (first word, in case there's additional info)
                if line:
                    ip = line.split()[0]
                    servers.append(ip)
        return servers
    
    def build_host_string(self, servers: List[str], num_nodes: int, gpus_per_node: int) -> str:
        """Build MPI host string for specified number of nodes.
        
        Args:
            servers: List of all available server IPs
            num_nodes: Number of nodes to use
            gpus_per_node: GPUs per node
            
        Returns:
            Host string like "ip1:8,ip2:8"
        """
        if num_nodes > len(servers):
            raise ValueError(f"Requested {num_nodes} nodes but only {len(servers)} servers available")
        
        selected = servers[:num_nodes]
        return ','.join(f"{ip}:{gpus_per_node}" for ip in selected)
    
    def expand_path(self, path: str) -> str:
        """Expand environment variables in path.
        
        Args:
            path: Path potentially containing ${VAR} references
            
        Returns:
            Expanded path
        """
        return os.path.expandvars(path)
    
    def build_mpirun_command(self, 
                            collective: str,
                            host_string: str,
                            num_gpus: int,
                            num_channels: Optional[int] = None,
                            test_params: Optional[Dict] = None,
                            algo: Optional[str] = None,
                            proto: Optional[str] = None) -> Tuple[List[str], Dict[str, str]]:
        """Build the mpirun command with all environment variables.
        
        Args:
            collective: Collective operation name (e.g., 'all_reduce_perf')
            host_string: MPI host specification
            num_gpus: Total number of GPUs (processes)
            num_channels: Number of channels to use, or None for NCCL default
            test_params: Optional override for test parameters
            algo: NCCL algorithm (Ring, Tree) or None for default
            proto: NCCL protocol (LL, LL128, SIMPLE) or None for default
            
        Returns:
            Tuple of (command list, environment dict)
        """
        mpi_config = self.config.get('mpi', {})
        test_defaults = self.config.get('test_defaults', {})
        env_vars = self.config.get('env_vars', {}).copy()
        paths = self.config.get('paths', {})
        
        # Override with any provided test params
        if test_params:
            test_defaults = {**test_defaults, **test_params}
        
        # Set channel counts only if specified
        if num_channels is not None:
            env_vars['NCCL_MIN_NCHANNELS'] = str(num_channels)
            env_vars['NCCL_MAX_NCHANNELS'] = str(num_channels)
        
        # Set algorithm if specified
        if algo is not None:
            env_vars['NCCL_ALGO'] = algo
        
        # Set protocol if specified
        if proto is not None:
            env_vars['NCCL_PROTO'] = proto
        
        # Get and expand the single rccl_path (contains both libs and executables)
        rccl_path = self.expand_path(paths.get('rccl_path', ''))
        
        if not rccl_path or rccl_path.startswith('$'):
            raise ValueError(
                f"rccl_path not set or not expanded: '{paths.get('rccl_path', '')}'. "
                "Set MY_PATH environment variable or provide absolute path in sweep_config.yaml"
            )
        
        # Set library paths using the single path
        env_vars['LD_LIBRARY_PATH'] = f"/usr/local/lib:{rccl_path}/:/opt/rocm/bin"
        # GPU-107: our bin/ has no librccl-net.so (built-in net_ib works on this fabric); preload only librccl.so
        env_vars['LD_PRELOAD'] = f"{rccl_path}/librccl.so"
        
        # Build base mpirun command
        mpirun_path = mpi_config.get('mpirun_path', '/opt/ompi-4.1.6/bin/mpirun')
        cmd = [
            mpirun_path,
            '--np', str(num_gpus),
            '--allow-run-as-root',
            '-H', host_string,
            '--bind-to', mpi_config.get('bind_to', 'numa'),
        ]
        
        # Add MPI interface settings
        oob_if = mpi_config.get('oob_tcp_if')
        btl_if = mpi_config.get('btl_tcp_if')
        if oob_if:
            cmd.extend(['--mca', 'oob_tcp_if_include', oob_if])
        if btl_if:
            cmd.extend(['--mca', 'btl_tcp_if_include', btl_if])
        
        # Exclude vader and openib btl modules
        cmd.extend(['--mca', 'btl', '^vader,openib'])
        
        # Add environment variables with -x flag
        for key, value in env_vars.items():
            cmd.extend(['-x', f'{key}={value}'])
        
        # Build test executable path (executables are in the same rccl_path)
        test_executable = os.path.join(rccl_path, collective)
        
        # Test arguments
        cmd.append(test_executable)
        cmd.extend(['-b', str(test_defaults.get('min_bytes', '1M'))])
        cmd.extend(['-e', str(test_defaults.get('max_bytes', '16G'))])
        # Use -i (step bytes) if step_bytes is set, otherwise use -f (step factor)
        if test_defaults.get('step_bytes'):
            cmd.extend(['-i', str(test_defaults.get('step_bytes'))])
        else:
            cmd.extend(['-f', str(test_defaults.get('step_factor', 2))])
        cmd.extend(['-g', str(test_defaults.get('gpus_per_rank', 1))])
        cmd.extend(['-n', str(test_defaults.get('iterations', 20))])
        cmd.extend(['-w', str(test_defaults.get('warmup_iters', 5))])
        cmd.extend(['-c', str(test_defaults.get('check_iters', 1))])
        cmd.extend(['-M', str(test_defaults.get('show_algo_proto_channels', 1))])
        
        if test_defaults.get('report_cputime'):
            cmd.extend(['-R', str(test_defaults.get('report_cputime', 1))])
        
        return cmd, env_vars
    
    def execute_test(self,
                    collective: str,
                    num_nodes: int,
                    num_gpus: int,
                    num_channels: Optional[int] = None,
                    host_string: str = "",
                    test_params: Optional[Dict] = None,
                    algo: Optional[str] = None,
                    proto: Optional[str] = None,
                    dry_run: bool = False) -> Dict[str, Any]:
        """Execute a single RCCL test.
        
        Args:
            collective: Collective operation name
            num_nodes: Number of nodes used
            num_gpus: Total number of GPUs
            num_channels: Number of channels, or None for NCCL default
            host_string: MPI host specification
            test_params: Optional test parameter overrides
            algo: NCCL algorithm (Ring, Tree) or None for default
            proto: NCCL protocol (LL, LL128, SIMPLE) or None for default
            dry_run: If True, just return the command without executing
            
        Returns:
            Dictionary with execution results
        """
        start_time = time.time()
        
        # Build command
        cmd, env_vars = self.build_mpirun_command(
            collective=collective,
            host_string=host_string,
            num_gpus=num_gpus,
            num_channels=num_channels,
            test_params=test_params,
            algo=algo,
            proto=proto
        )
        
        cmd_str = ' '.join(shlex.quote(c) if ' ' in c else c for c in cmd)
        
        # Create output directory for this run (include channels/algo/proto in name if set)
        run_name = f"{collective}_{num_nodes}node"
        if num_channels:
            run_name += f"_{num_channels}ch"
        if algo:
            run_name += f"_{algo}"
        if proto:
            run_name += f"_{proto}"
        run_dir = self.output_dir / run_name
        run_dir.mkdir(exist_ok=True)
        
        # Save command
        cmd_file = run_dir / "command.txt"
        with open(cmd_file, 'w') as f:
            f.write(cmd_str + '\n')
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'collective': collective,
            'num_nodes': num_nodes,
            'num_gpus': num_gpus,
            'num_channels': num_channels,
            'command': cmd_str,
            'output_path': str(run_dir),
            'status': 'pending',
            'stdout': '',
            'stderr': '',
            'return_code': None,
            'duration_sec': 0
        }
        
        if dry_run:
            result['status'] = 'dry_run'
            return result
        
        if self.verbose:
            ch_str = f" | {num_channels} channels" if num_channels else ""
            algo_str = f" | algo={algo}" if algo else ""
            proto_str = f" | proto={proto}" if proto else ""
            print(f"\n{'─'*60}")
            print(f"Running: {collective} | {num_nodes} node(s){ch_str}{algo_str}{proto_str}")
            print(f"{'─'*60}")
        
        # Execute command
        try:
            timeout = self.config.get('test_defaults', {}).get('timeout', 600)
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy()
            )
            
            stdout, stderr = process.communicate(timeout=timeout)
            return_code = process.returncode
            
            duration = time.time() - start_time
            
            result.update({
                'stdout': stdout,
                'stderr': stderr,
                'return_code': return_code,
                'duration_sec': duration,
                'status': 'success' if return_code == 0 else 'failed'
            })
            
            # Save output
            output_file = run_dir / "output.log"
            with open(output_file, 'w') as f:
                f.write(stdout)
            
            if stderr:
                error_file = run_dir / "error.log"
                with open(error_file, 'w') as f:
                    f.write(stderr)
            
            if self.verbose:
                if return_code == 0:
                    print(f"✓ Completed in {duration:.1f}s")
                else:
                    print(f"✗ Failed with code {return_code} in {duration:.1f}s")
            
        except subprocess.TimeoutExpired:
            process.kill()
            duration = time.time() - start_time
            
            result.update({
                'status': 'timeout',
                'duration_sec': duration,
                'stderr': f'Timeout after {timeout}s'
            })
            
            if self.verbose:
                print(f"✗ Timeout after {duration:.1f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            
            result.update({
                'status': 'error',
                'duration_sec': duration,
                'stderr': str(e)
            })
            
            if self.verbose:
                print(f"✗ Error: {e}")
        
        return result
    
    def get_output_dir(self) -> Path:
        """Get the output directory path."""
        return self.output_dir


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


if __name__ == '__main__':
    import yaml
    
    # Test with sample config
    sample_config = {
        'mpi': {
            'bind_to': 'numa',
            'oob_tcp_if': 'enp81s0f1np1',
            'btl_tcp_if': 'enp81s0f1np1',
        },
        'test_defaults': {
            'min_bytes': '1M',
            'max_bytes': '16G',
            'step_factor': 2,
            'gpus_per_rank': 1,
            'iterations': 20,
            'warmup_iters': 5,
            'timeout': 600,
        },
        'paths': {
            # Single path containing both RCCL libs and rccl-tests executables
            'rccl_path': '/tmp/test_rccl_path',
        },
        'env_vars': {
            'NCCL_IB_HCA': 'ionic_0:1,ionic_1:1',
            'NCCL_DEBUG': 'VERSION',
        }
    }
    
    executor = SweepExecutor(sample_config, output_dir='/tmp/sweep_test')
    
    # Test command building (dry run)
    cmd, env = executor.build_mpirun_command(
        collective='all_reduce_perf',
        host_string='192.168.1.1:8,192.168.1.2:8',
        num_gpus=16,
        num_channels=32
    )
    
    print("Generated command:")
    print(' '.join(cmd))

