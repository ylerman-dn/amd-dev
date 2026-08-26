#!/usr/bin/env python3
"""
Parser for RCCL test output to extract performance metrics.
Extracts bandwidth metrics for each message size from rccl-tests output.
"""

import re
import json
from typing import Dict, List, Optional, Any
from collections import Counter


class RCCLOutputParser:
    """Parses RCCL test output to extract performance metrics."""
    
    # Standard collectives and their test executable names
    COLLECTIVES = {
        'all_reduce': 'all_reduce_perf',
        'reduce_scatter': 'reduce_scatter_perf',
        'all_gather': 'all_gather_perf',
        'alltoall': 'alltoall_perf',
        'broadcast': 'broadcast_perf',
        'reduce': 'reduce_perf',
    }
    
    @staticmethod
    def parse_output(stdout: str) -> Dict[str, Any]:
        """Parse RCCL test output and extract metrics.
        
        Args:
            stdout: Complete stdout from RCCL test
            
        Returns:
            Dictionary with parsed metrics including:
            - success: bool
            - collective: str (detected collective name)
            - rccl_version: str
            - metrics: list of dicts per message size
            - avg_busbw: float
            - error_message: str (if failed)
        """
        result = {
            'success': False,
            'collective': None,
            'rccl_version': None,
            'hip_version': None,
            'rocm_version': None,
            'num_ranks': 0,
            'devices': [],
            'metrics': [],
            'avg_busbw': None,
            'max_busbw': None,
            'detected_algo': None,      # Most common algo from output
            'detected_proto': None,     # Most common proto from output
            'detected_nchannels': None, # nchannels from output (typically constant)
            'error_message': None
        }
        
        lines = stdout.split('\n')
        
        # Extract collective name from header
        for line in lines:
            if 'Collective test starting:' in line:
                match = re.search(r'Collective test starting:\s+(\w+)', line)
                if match:
                    result['collective'] = match.group(1)
                break
        
        # Extract version info
        for line in lines:
            if 'RCCL version' in line:
                match = re.search(r'RCCL version\s*:\s*(.+)', line)
                if match:
                    result['rccl_version'] = match.group(1).strip()
            elif 'HIP version' in line:
                match = re.search(r'HIP version\s*:\s*(.+)', line)
                if match:
                    result['hip_version'] = match.group(1).strip()
            elif 'ROCm version' in line:
                match = re.search(r'ROCm version\s*:\s*(.+)', line)
                if match:
                    result['rocm_version'] = match.group(1).strip()
        
        # Extract device info and count ranks
        rank_pattern = re.compile(r'#\s+Rank\s+(\d+)\s+Group\s+(\d+)\s+Pid\s+(\d+)\s+on\s+(\S+)\s+device\s+(\d+)\s+\[([^\]]+)\]\s+(.+)')
        for line in lines:
            match = rank_pattern.search(line)
            if match:
                result['devices'].append({
                    'rank': int(match.group(1)),
                    'group': int(match.group(2)),
                    'pid': int(match.group(3)),
                    'hostname': match.group(4),
                    'device_id': int(match.group(5)),
                    'pci_bus': match.group(6),
                    'device_name': match.group(7).strip()
                })
        
        result['num_ranks'] = len(result['devices'])
        
        # Parse the results table
        # Format: size count type redop root time algbw busbw #wrong time algbw busbw #wrong
        metrics = []
        in_results = False
        
        for line in lines:
            stripped = line.strip()
            
            # Detect start of results section
            if '#       size' in line and 'count' in line and 'type' in line:
                in_results = True
                continue
            
            # Skip header lines and comments
            if not stripped or stripped.startswith('#'):
                # Check for average bus bandwidth in comments
                if 'Avg bus bandwidth' in stripped:
                    match = re.search(r'Avg bus bandwidth\s*:\s*([\d.]+)', stripped)
                    if match:
                        result['avg_busbw'] = float(match.group(1))
                continue
            
            # Stop at end markers
            if 'Out of bounds values' in stripped or 'Collective test concluded' in stripped:
                continue
            
            if in_results:
                try:
                    parts = stripped.split()
                    if len(parts) >= 13:
                        # Helper to parse int that may be 'N/A'
                        def parse_int_or_na(val):
                            return -1 if val == 'N/A' else int(val)
                        
                        metric = {
                            'size_bytes': int(parts[0]),
                            'count': int(parts[1]),
                            'type': parts[2],
                            'redop': parts[3],
                            'root': int(parts[4]),
                            # Out-of-place metrics
                            'time_oop_us': float(parts[5]),
                            'algbw_oop': float(parts[6]),
                            'busbw_oop': float(parts[7]),
                            'errors_oop': parse_int_or_na(parts[8]),
                            # In-place metrics
                            'time_ip_us': float(parts[9]),
                            'algbw_ip': float(parts[10]),
                            'busbw_ip': float(parts[11]),
                            'errors_ip': parse_int_or_na(parts[12]),
                        }
                        # Parse extended columns if present (algo, proto, nchannels).
                        # alltoall runs on the p2p path (send/recv tasks, no collective
                        # plan), so -A 1 prints N/A in all three columns; the row is
                        # still a valid measurement and must not be dropped.
                        if len(parts) >= 16:
                            if parts[13] != 'N/A':
                                metric['algo'] = parts[13]
                            if parts[14] != 'N/A':
                                metric['proto'] = parts[14]
                            if parts[15] != 'N/A':
                                metric['nchannels'] = int(parts[15])
                        metrics.append(metric)
                except (ValueError, IndexError):
                    # Skip lines that don't match expected format
                    continue
        
        result['metrics'] = metrics
        
        # Extract detected algo/proto/nchannels from metrics
        if metrics:
            # Get nchannels (should be constant across all rows)
            nchannels_values = [m.get('nchannels') for m in metrics if m.get('nchannels') is not None]
            if nchannels_values:
                result['detected_nchannels'] = nchannels_values[0]  # Use first value (should all be same)
            
            # Get most common algo (may vary by message size, e.g., Direct vs RING)
            algo_values = [m.get('algo') for m in metrics if m.get('algo') is not None]
            if algo_values:
                result['detected_algo'] = Counter(algo_values).most_common(1)[0][0]
            
            # Get most common proto
            proto_values = [m.get('proto') for m in metrics if m.get('proto') is not None]
            if proto_values:
                result['detected_proto'] = Counter(proto_values).most_common(1)[0][0]
        
        if metrics:
            result['success'] = True
            # Calculate max bus bandwidth
            busbw_values = [m['busbw_oop'] for m in metrics]
            result['max_busbw'] = max(busbw_values) if busbw_values else None
        
        # Check for errors
        if not result['success']:
            error_patterns = [
                r'(Error|ERROR|FAILED).*',
                r'Segmentation fault',
                r'Bus error',
                r'HIP error',
                r'RCCL error',
            ]
            for pattern in error_patterns:
                match = re.search(pattern, stdout, re.IGNORECASE)
                if match:
                    result['error_message'] = match.group(0)[:200]
                    break
            
            if not result['error_message']:
                result['error_message'] = 'No valid metrics found in output'
        
        return result
    
    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Format byte size to human readable string."""
        if size_bytes >= 1024**3:
            return f"{size_bytes / 1024**3:.0f}G"
        elif size_bytes >= 1024**2:
            return f"{size_bytes / 1024**2:.0f}M"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.0f}K"
        else:
            return f"{size_bytes}B"
    
    @staticmethod
    def metrics_to_json(metrics: List[Dict]) -> str:
        """Convert metrics list to JSON string for database storage."""
        return json.dumps(metrics)
    
    @staticmethod
    def json_to_metrics(json_str: str) -> List[Dict]:
        """Convert JSON string back to metrics list."""
        return json.loads(json_str)


def parse_rccl_output(stdout: str, stderr: str = "") -> Dict[str, Any]:
    """High-level function to parse RCCL test output.
    
    Args:
        stdout: Standard output from test
        stderr: Standard error from test (optional)
        
    Returns:
        Dictionary with parsed results
    """
    parser = RCCLOutputParser()
    result = parser.parse_output(stdout)
    
    # Check stderr for additional errors
    if stderr and not result['success']:
        if 'error' in stderr.lower() or 'failed' in stderr.lower():
            if not result['error_message']:
                result['error_message'] = stderr[:200]
    
    return result


if __name__ == '__main__':
    # Test with sample output (new format with algo/proto/nchannels)
    sample_output = """# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1048576 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
rccl-tests: Version develop:6405c76
# Using devices
#  Rank  0 Group  0 Pid 3252337 on amd-mi350x-ses2-1 device  0 [0000:05:00] AMD Instinct MI350X
#  Rank  1 Group  0 Pid 3252338 on amd-mi350x-ses2-1 device  1 [0000:15:00] AMD Instinct MI350X
RCCL version : 2.27.7-develop:29e1567
HIP version  : 7.0.51831-a3e329ad8
ROCm version : 7.0.1.0-42-9428210
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong     algo     proto   nchannels
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)                                      
     1048576        262144     float     sum      -1    76.26   13.75   25.78      0    76.43   13.72   25.72      0  Direct    SIMPLE          32
     2097152        524288     float     sum      -1    87.02   24.10   45.18      0    86.52   24.24   45.45      0    RING    SIMPLE          32
   268435456      67108864     float     sum      -1   1377.7  194.84  365.33      0   1376.1  195.07  365.76      0    RING    SIMPLE          32
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 249.115 
#
# Collective test concluded: all_reduce_perf
"""
    
    result = parse_rccl_output(sample_output)
    print(f"Success: {result['success']}")
    print(f"Collective: {result['collective']}")
    print(f"RCCL Version: {result['rccl_version']}")
    print(f"Num Ranks: {result['num_ranks']}")
    print(f"Avg Bus BW: {result['avg_busbw']} GB/s")
    print(f"Max Bus BW: {result['max_busbw']} GB/s")
    print(f"Detected Algo: {result['detected_algo']}")
    print(f"Detected Proto: {result['detected_proto']}")
    print(f"Detected nChannels: {result['detected_nchannels']}")
    print(f"Metrics count: {len(result['metrics'])}")
    for m in result['metrics']:
        algo_info = f" [{m.get('algo', 'N/A')}/{m.get('proto', 'N/A')}/{m.get('nchannels', 'N/A')}ch]" if 'algo' in m else ""
        print(f"  {RCCLOutputParser.format_size(m['size_bytes'])}: {m['busbw_oop']:.2f} GB/s{algo_info}")

