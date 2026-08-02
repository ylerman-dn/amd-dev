#!/usr/bin/env python3
"""
RCCL Test Sweep Tool

Systematically runs RCCL collective tests across multiple configurations:
- Multiple collective operations (all_reduce, alltoall, etc.)
- Node scaling (1-N nodes)
- Channel sweep (configurable range and step)

Results are stored in SQLite database with full command and output logging.
"""

import argparse
import sys
import os
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from colorama import init, Fore, Style
from tabulate import tabulate

# Initialize colorama
init()

from sweep_executor import SweepExecutor, format_duration
from sweep_parser import parse_rccl_output, RCCLOutputParser
from sweep_db import SweepDatabase


# Default collectives
DEFAULT_COLLECTIVES = [
    'all_reduce_perf',
    'reduce_scatter_perf', 
    'all_gather_perf',
    'alltoall_perf',
    'broadcast_perf',
    'reduce_perf',
]


def parse_nodes(nodes_str: str, max_available: int) -> List[int]:
    """Parse nodes specification string.
    
    Args:
        nodes_str: Format "N" for single node count, or "MIN-MAX" for range
                   e.g., "2" for 2 nodes, "1-4" for 1,2,3,4 nodes
        max_available: Maximum number of available servers
        
    Returns:
        List of node counts
    """
    if '-' in nodes_str:
        # Range format: "1-4"
        parts = nodes_str.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid nodes format '{nodes_str}'. Use N or MIN-MAX (e.g., 2 or 1-4)")
        try:
            min_nodes = int(parts[0])
            max_nodes = int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid nodes format '{nodes_str}'. Values must be integers.")
        
        if min_nodes < 1 or max_nodes < min_nodes:
            raise ValueError(f"Invalid node range: min={min_nodes}, max={max_nodes}")
        
        if max_nodes > max_available:
            raise ValueError(f"Requested up to {max_nodes} nodes but only {max_available} servers available")
        
        return list(range(min_nodes, max_nodes + 1))
    else:
        # Single value: "2"
        try:
            num_nodes = int(nodes_str)
        except ValueError:
            raise ValueError(f"Invalid nodes format '{nodes_str}'. Use N or MIN-MAX (e.g., 2 or 1-4)")
        
        if num_nodes < 1:
            raise ValueError(f"Invalid node count: {num_nodes}")
        
        if num_nodes > max_available:
            raise ValueError(f"Requested {num_nodes} nodes but only {max_available} servers available")
        
        return [num_nodes]


def parse_channels(channels_str: str) -> List[int]:
    """Parse channels specification string.
    
    Args:
        channels_str: Either:
            - Range format: "MIN:MAX:STEP" (e.g., "4:64:4")
            - List format: "V1,V2,V3,..." (e.g., "1,5,16,32,64,128")
            - Single value: "32"
        
    Returns:
        List of channel values
    """
    # Check for comma-separated list format
    if ',' in channels_str:
        try:
            channels = [int(v.strip()) for v in channels_str.split(',')]
        except ValueError:
            raise ValueError(f"Invalid channels list '{channels_str}'. Values must be integers.")
        
        # Validate values
        for ch in channels:
            if ch < 1:
                raise ValueError(f"Invalid channel value: {ch}. Must be >= 1")
        
        return sorted(channels)
    
    # Check for range format (contains :)
    if ':' in channels_str:
        parts = channels_str.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid channels format '{channels_str}'. Use MIN:MAX:STEP (e.g., 4:64:4)")
        
        try:
            min_ch = int(parts[0])
            max_ch = int(parts[1])
            step = int(parts[2])
        except ValueError:
            raise ValueError(f"Invalid channels format '{channels_str}'. Values must be integers.")
        
        if min_ch < 1 or max_ch < min_ch or step < 1:
            raise ValueError(f"Invalid channel range: min={min_ch}, max={max_ch}, step={step}")
        
        return list(range(min_ch, max_ch + 1, step))
    
    # Single value
    try:
        ch = int(channels_str.strip())
    except ValueError:
        raise ValueError(f"Invalid channels format '{channels_str}'. Use MIN:MAX:STEP, comma-separated list, or single value.")
    
    if ch < 1:
        raise ValueError(f"Invalid channel value: {ch}. Must be >= 1")
    
    return [ch]


# Algorithm and protocol mappings
ALGO_MAP = {
    'RING': 'Ring',
    'TREE': 'Tree',
    'DIRECT': 'Direct',
}

PROTO_MAP = {
    'LL': 'LL',
    'LL128': 'LL128',
    'SIMPLE': 'SIMPLE',
}


def parse_algo(algo_str: Optional[str]) -> List[Optional[str]]:
    """Parse algorithm specification.
    
    Args:
        algo_str: 'all', comma-separated list (e.g., 'RING,TREE'), single value, or None
        
    Returns:
        List of algorithm values (None means use NCCL default)
    """
    if algo_str is None:
        return [None]
    if algo_str == 'all':
        return list(ALGO_MAP.keys())
    
    # Parse comma-separated list
    algos = []
    for algo in algo_str.split(','):
        algo = algo.strip().upper()
        if algo not in ALGO_MAP:
            print(f"{Fore.RED}Error: Invalid algorithm '{algo}'. "
                  f"Valid options: {', '.join(sorted(ALGO_MAP.keys()))}{Style.RESET_ALL}")
            sys.exit(1)
        algos.append(algo)
    return algos


def parse_proto(proto_str: Optional[str]) -> List[Optional[str]]:
    """Parse protocol specification.
    
    Args:
        proto_str: 'all', comma-separated list (e.g., 'LL,LL128'), single value, or None
        
    Returns:
        List of protocol values (None means use NCCL default)
    """
    if proto_str is None:
        return [None]
    if proto_str == 'all':
        return list(PROTO_MAP.keys())
    
    # Parse comma-separated list
    protos = []
    for proto in proto_str.split(','):
        proto = proto.strip().upper()
        if proto not in PROTO_MAP:
            print(f"{Fore.RED}Error: Invalid protocol '{proto}'. "
                  f"Valid options: {', '.join(sorted(PROTO_MAP.keys()))}{Style.RESET_ALL}")
            sys.exit(1)
        protos.append(proto)
    return protos


def get_algo_env_value(algo: Optional[str]) -> Optional[str]:
    """Get NCCL_ALGO environment variable value.
    
    Args:
        algo: Algorithm name (RING, TREE) or None
        
    Returns:
        Value for NCCL_ALGO or None if not set
    """
    if algo is None:
        return None
    return ALGO_MAP.get(algo)


def get_proto_env_value(proto: Optional[str]) -> Optional[str]:
    """Get NCCL_PROTO environment variable value.
    
    Args:
        proto: Protocol name (LL, LL128, SIMPLE) or None
        
    Returns:
        Value for NCCL_PROTO or None if not set
    """
    if proto is None:
        return None
    return PROTO_MAP.get(proto)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_header():
    """Print sweep tool header."""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Style.BRIGHT}  RCCL Test Sweep Tool{Style.RESET_ALL}")
    print(f"{'='*70}{Style.RESET_ALL}\n")


def print_sweep_plan(collectives: List[str], 
                     num_nodes_list: List[int],
                     channels: List[Optional[int]],
                     gpus_per_node: int,
                     algos: List[Optional[str]],
                     protos: List[Optional[str]]):
    """Print the planned test matrix."""
    total_tests = len(collectives) * len(num_nodes_list) * len(channels) * len(algos) * len(protos)
    
    print(f"{Fore.GREEN}Sweep Configuration:{Style.RESET_ALL}")
    print(f"  Collectives: {', '.join(c.replace('_perf', '') for c in collectives)}")
    print(f"  Node counts: {', '.join(str(n) for n in num_nodes_list)} ({gpus_per_node} GPUs/node)")
    
    # Display channels configuration
    if channels == [None]:
        print(f"  Channels: default (NCCL auto)")
    else:
        print(f"  Channels: {min(channels)}-{max(channels)} (step {channels[1]-channels[0] if len(channels)>1 else 'N/A'})")
    
    # Display algo/proto configuration
    if algos == [None]:
        print(f"  Algorithms: default (NCCL auto)")
    else:
        print(f"  Algorithms: {', '.join(algos)}")
    
    if protos == [None]:
        print(f"  Protocols: default (NCCL auto)")
    else:
        print(f"  Protocols: {', '.join(protos)}")
    
    print(f"  Total tests: {Fore.YELLOW}{total_tests}{Style.RESET_ALL}")
    
    # Estimate duration
    est_per_test = 180  # ~3 minutes per test
    est_total = total_tests * est_per_test
    print(f"  Estimated time: {Fore.YELLOW}{format_duration(est_total)}{Style.RESET_ALL}")
    print()


def run_sweep(args, config: Dict[str, Any]):
    """Run the test sweep.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
    """
    # Read servers
    if not os.path.exists(args.servers):
        print(f"{Fore.RED}Error: Servers file not found: {args.servers}{Style.RESET_ALL}")
        sys.exit(1)
    
    executor = SweepExecutor(config, output_dir='/tmp/sweep_temp', verbose=False)
    servers = executor.read_servers(args.servers)
    
    if not servers:
        print(f"{Fore.RED}Error: No servers found in {args.servers}{Style.RESET_ALL}")
        sys.exit(1)
    
    print(f"Found {len(servers)} server(s) in {args.servers}")
    
    # Determine collectives to run
    valid_collectives = {'all_reduce', 'reduce_scatter', 'all_gather', 'alltoall', 'broadcast', 'reduce'}
    
    if args.collective:
        if args.collective == 'all':
            collectives = config.get('collectives', DEFAULT_COLLECTIVES)
        else:
            # Parse comma-separated list
            collectives = []
            for coll in args.collective.split(','):
                coll = coll.strip()
                # Strip _perf suffix for validation
                coll_base = coll.replace('_perf', '')
                if coll_base not in valid_collectives:
                    print(f"{Fore.RED}Error: Invalid collective '{coll}'. "
                          f"Valid options: {', '.join(sorted(valid_collectives))}{Style.RESET_ALL}")
                    sys.exit(1)
                # Add _perf suffix if not present
                if not coll.endswith('_perf'):
                    coll = f"{coll}_perf"
                collectives.append(coll)
    else:
        collectives = config.get('collectives', DEFAULT_COLLECTIVES)
    
    # Determine node counts
    gpus_per_node = config.get('nodes', {}).get('gpus_per_node', 8)
    max_available = min(len(servers), 9)  # Cap at 9 as per requirements
    
    if args.nodes:
        try:
            num_nodes_list = parse_nodes(args.nodes, max_available)
        except ValueError as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            sys.exit(1)
    else:
        # Use all available nodes (1 to N)
        num_nodes_list = list(range(1, max_available + 1))
    
    # Parse channels (optional - use [None] if not provided to use NCCL defaults)
    if args.channels:
        channels = parse_channels(args.channels)
    else:
        channels = [None]
    
    # Parse algorithm and protocol options
    algos = parse_algo(getattr(args, 'algo', None))
    protos = parse_proto(getattr(args, 'proto', None))
    
    # Print plan
    print_sweep_plan(collectives, num_nodes_list, channels, gpus_per_node, algos, protos)
    
    # Dry run mode
    if args.dry_run:
        print(f"{Fore.YELLOW}DRY RUN MODE - Commands will be shown but not executed{Style.RESET_ALL}\n")
        
        for collective in collectives:
            for num_nodes in num_nodes_list:
                num_gpus = num_nodes * gpus_per_node
                host_string = executor.build_host_string(servers, num_nodes, gpus_per_node)
                
                for num_channels in channels:
                    for algo in algos:
                        for proto in protos:
                            cmd, _ = executor.build_mpirun_command(
                                collective=collective,
                                host_string=host_string,
                                num_gpus=num_gpus,
                                num_channels=num_channels,
                                algo=get_algo_env_value(algo),
                                proto=get_proto_env_value(proto)
                            )
                            
                            # Build display string for channels/algo/proto
                            ch_str = f", {num_channels} ch" if num_channels else ""
                            algo_str = f", algo={algo}" if algo else ""
                            proto_str = f", proto={proto}" if proto else ""
                            
                            print(f"{Fore.CYAN}[{collective}] {num_nodes} node(s){ch_str}{algo_str}{proto_str}:{Style.RESET_ALL}")
                            print(f"  {' '.join(cmd)}")
                            print()
        
        total = len(collectives) * len(num_nodes_list) * len(channels) * len(algos) * len(protos)
        print(f"Total: {total} tests would be executed")
        return
    
    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = args.output_dir or config.get('output', {}).get('results_dir', 'sweep_results')
    output_dir = Path(base_output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize executor with real output dir
    executor = SweepExecutor(config, output_dir=str(output_dir / 'outputs'), verbose=True)
    
    # Initialize database
    db_path = output_dir / config.get('output', {}).get('database', 'sweep_results.db')
    db = SweepDatabase(str(db_path))
    
    # Calculate total tests
    total_tests = len(collectives) * len(num_nodes_list) * len(channels) * len(algos) * len(protos)
    
    # Create session
    session_config = {
        'collectives': collectives,
        'num_nodes_list': num_nodes_list,
        'channels': channels,
        'algos': algos,
        'protos': protos,
        'servers': servers[:max(num_nodes_list)],
        'config': config
    }
    session_id = db.create_session(session_config, total_tests)
    
    print(f"{Fore.GREEN}Output directory: {output_dir}{Style.RESET_ALL}")
    print(f"Database: {db_path}")
    print(f"Session ID: {session_id}")
    print()
    
    # Run sweep
    completed = 0
    failed = 0
    start_time = time.time()
    
    try:
        for collective in collectives:
            for num_nodes in num_nodes_list:
                num_gpus = num_nodes * gpus_per_node
                host_string = executor.build_host_string(servers, num_nodes, gpus_per_node)
                
                for num_channels in channels:
                    for algo in algos:
                        for proto in protos:
                            completed += 1
                            
                            # Progress header
                            progress = f"[{completed}/{total_tests}]"
                            elapsed = time.time() - start_time
                            if completed > 1:
                                avg_time = elapsed / (completed - 1)
                                remaining = avg_time * (total_tests - completed + 1)
                                eta = f"ETA: {format_duration(remaining)}"
                            else:
                                eta = ""
                            
                            # Build display string for channels/algo/proto
                            ch_str = f" | {num_channels} ch" if num_channels else ""
                            algo_str = f" | {algo}" if algo else ""
                            proto_str = f" | {proto}" if proto else ""
                            
                            print(f"\n{Fore.CYAN}{progress} {collective} | {num_nodes} node(s){ch_str}{algo_str}{proto_str} {eta}{Style.RESET_ALL}")
                            
                            # Get env var values for algo/proto
                            algo_env = get_algo_env_value(algo)
                            proto_env = get_proto_env_value(proto)
                            
                            # Execute test
                            result = executor.execute_test(
                                collective=collective,
                                num_nodes=num_nodes,
                                num_gpus=num_gpus,
                                num_channels=num_channels,
                                host_string=host_string,
                                algo=algo_env,
                                proto=proto_env,
                                dry_run=False
                            )
                            
                            # Parse output
                            parsed = parse_rccl_output(result['stdout'], result['stderr'])
                            
                            # Use detected nchannels from output (ground truth), fall back to requested if not available
                            # -1 indicates no valid channels value was detected (invalid/error condition)
                            actual_nchannels = parsed.get('detected_nchannels') or num_channels or -1
                            
                            # Prepare database entry
                            db_entry = {
                                'session_id': session_id,
                                'timestamp': result['timestamp'],
                                'collective': collective,
                                'num_nodes': num_nodes,
                                'num_gpus': num_gpus,
                                'num_channels': actual_nchannels,
                                'algo': algo,
                                'proto': proto,
                                'command': result['command'],
                                'status': result['status'],
                                'duration_sec': result['duration_sec'],
                                'output_path': result['output_path'],
                            }
                            
                            if parsed['success']:
                                db_entry.update({
                                    'results_json': RCCLOutputParser.metrics_to_json(parsed['metrics']),
                                    'avg_busbw': parsed['avg_busbw'],
                                    'max_busbw': parsed['max_busbw'],
                                    'rccl_version': parsed['rccl_version'],
                                    'hip_version': parsed['hip_version'],
                                    'rocm_version': parsed['rocm_version'],
                                })
                                
                                print(f"  Avg BW: {Fore.GREEN}{parsed['avg_busbw']:.2f} GB/s{Style.RESET_ALL}, "
                                      f"Max: {parsed['max_busbw']:.2f} GB/s, nCh: {actual_nchannels}")
                            else:
                                db_entry['error_message'] = parsed.get('error_message', result.get('stderr', ''))[:500]
                                failed += 1
                            
                            # Save to database
                            run_id = db.insert_run(db_entry)
                            
                            # Insert per-message-size metrics (granular data)
                            if parsed['success'] and parsed.get('metrics'):
                                # GPU-107: our rccl-tests build emits no -M algo/proto/channel columns,
                                # so the parser can't detect them. Stamp each per-size row with the
                                # run's known forced values so optimize/generate steps have them.
                                for _m in parsed['metrics']:
                                    if not _m.get('nchannels'):
                                        _m['nchannels'] = actual_nchannels
                                    if not _m.get('algo'):
                                        _m['algo'] = algo
                                    if not _m.get('proto'):
                                        _m['proto'] = proto
                                db.insert_metrics(run_id, parsed['metrics'])
        
        # Complete session
        db.complete_session(session_id)
        
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Sweep interrupted by user{Style.RESET_ALL}")
        db.update_session(session_id, status='interrupted', end_time=datetime.now().isoformat())
    
    finally:
        # Print summary
        elapsed_total = time.time() - start_time
        
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Style.BRIGHT}Sweep Complete{Style.RESET_ALL}")
        print(f"{'='*70}{Style.RESET_ALL}\n")
        
        print(f"  Total time: {format_duration(elapsed_total)}")
        print(f"  Tests run: {completed}")
        print(f"  Successful: {Fore.GREEN}{completed - failed}{Style.RESET_ALL}")
        print(f"  Failed: {Fore.RED}{failed}{Style.RESET_ALL}")
        print(f"  Success rate: {(completed - failed) / completed * 100:.1f}%")
        print()
        print(f"  Results: {output_dir}")
        print(f"  Database: {db_path}")
        
        # Export CSV summary
        csv_path = output_dir / 'summary.csv'
        db.export_to_csv(str(csv_path), session_id)
        print(f"  CSV export: {csv_path}")
        
        # Export detailed per-size metrics
        metrics_csv_path = output_dir / 'metrics.csv'
        db.export_metrics_to_csv(str(metrics_csv_path), session_id)
        print(f"  Metrics CSV: {metrics_csv_path}")
        
        db.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='RCCL Test Sweep Tool - Systematically run RCCL collective tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with NCCL default channels (no channel sweep)
  %(prog)s --algo RING --proto LL128

  # Run all collectives, all nodes, channel sweep 4-64 step 4
  %(prog)s --channels 4:64:4

  # Run with explicit channel values
  %(prog)s --channels 1,5,16,32,64,128

  # Run single collective
  %(prog)s --collective all_reduce --channels 4:64:4

  # Run multiple collectives
  %(prog)s --collective all_reduce,all_gather --channels 4:64:4

  # Run on specific node count
  %(prog)s --nodes 2 --channels 4:64:4

  # Run on node range (1 to 4 nodes)
  %(prog)s --nodes 1-4 --channels 4:64:4

  # Sweep all algorithms with default protocol
  %(prog)s --channels 4:64:4 --algo all

  # Use specific algorithm and protocol
  %(prog)s --channels 4:64:4 --algo RING --proto LL128

  # Use multiple algorithms and/or protocols
  %(prog)s --channels 4:64:4 --algo RING,TREE --proto LL,LL128

  # Full sweep: all algos x all protos
  %(prog)s --channels 4:64:4 --algo all --proto all

  # Custom message sizes (default: 1M to 16G)
  %(prog)s --channels 4:64:4 --min-size 256M --max-size 1G

  # Dry run - show commands without executing
  %(prog)s --channels 4:64:4 --dry-run

  # Use custom servers file
  %(prog)s --servers /path/to/servers.txt --channels 4:64:4
"""
    )
    
    parser.add_argument(
        '--servers', '-s',
        default='./servers.txt',
        help='Path to servers.txt file with node IPs (default: ./servers.txt)'
    )
    
    parser.add_argument(
        '--channels', '-c',
        help='Channels: range MIN:MAX:STEP (e.g., 4:64:4) or list V1,V2,... (e.g., 1,5,16,32,64,128). If not provided, uses NCCL defaults'
    )
    
    parser.add_argument(
        '--collective',
        help='Collective(s) to run: comma-separated list (e.g., all_reduce,all_gather) or "all". '
             'Valid: all_reduce, reduce_scatter, all_gather, alltoall, broadcast, reduce'
    )
    
    parser.add_argument(
        '--nodes', '-n',
        help='Node count or range: N or MIN-MAX (e.g., 2 or 1-4). Default: all available'
    )
    
    parser.add_argument(
        '--algo',
        help='Algorithm(s) to use: comma-separated list (e.g., RING,TREE) or "all". '
             'Valid: RING, TREE, DIRECT. Default: NCCL auto'
    )
    
    parser.add_argument(
        '--proto',
        help='Protocol(s) to use: comma-separated list (e.g., LL,LL128) or "all". '
             'Valid: LL, LL128, SIMPLE. Default: NCCL auto'
    )
    
    parser.add_argument(
        '--config',
        default='sweep_config.yaml',
        help='Path to configuration file (default: sweep_config.yaml)'
    )
    
    parser.add_argument(
        '--min-size',
        help='Override minimum message size (e.g., 1M, 256M)'
    )
    
    parser.add_argument(
        '--max-size',
        help='Override maximum message size (e.g., 1G, 16G)'
    )
    
    parser.add_argument(
        '--step-size',
        help='Override step factor with fixed step size (e.g., 1M, 4K, 100000). '
             'Uses -i instead of default -f 2'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory for results (default: from config or sweep_results)'
    )
    
    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='Show commands without executing'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    print_header()
    
    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        # Try looking in script directory
        script_dir = Path(__file__).parent
        config_path = script_dir / 'sweep_config.yaml'
        if not config_path.exists():
            print(f"{Fore.RED}Error: Config file not found: {args.config}{Style.RESET_ALL}")
            sys.exit(1)
    
    config = load_config(str(config_path))
    
    # Apply overrides
    if args.min_size:
        config.setdefault('test_defaults', {})['min_bytes'] = args.min_size
    if args.max_size:
        config.setdefault('test_defaults', {})['max_bytes'] = args.max_size
    if args.step_size:
        config.setdefault('test_defaults', {})['step_bytes'] = args.step_size
    
    # Run sweep
    run_sweep(args, config)


if __name__ == '__main__':
    main()

