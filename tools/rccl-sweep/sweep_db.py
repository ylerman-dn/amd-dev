#!/usr/bin/env python3
"""
Database operations for storing and retrieving RCCL sweep test results.
Uses SQLite for portability and simplicity.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class SweepDatabase:
    """Manages storage and retrieval of sweep test results."""
    
    def __init__(self, db_path: str):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Main sweep session table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sweep_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                config_json TEXT NOT NULL,
                total_tests INTEGER DEFAULT 0,
                completed_tests INTEGER DEFAULT 0,
                failed_tests INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running'
            )
        """)
        
        # Individual test runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sweep_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                
                -- Test configuration
                collective TEXT NOT NULL,
                num_nodes INTEGER NOT NULL,
                num_gpus INTEGER NOT NULL,
                num_channels INTEGER NOT NULL,
                algo TEXT,
                proto TEXT,
                
                -- Command info
                command TEXT NOT NULL,
                
                -- Results
                results_json TEXT,
                avg_busbw REAL,
                max_busbw REAL,
                
                -- Version info
                rccl_version TEXT,
                hip_version TEXT,
                rocm_version TEXT,
                
                -- Execution info
                status TEXT NOT NULL,
                duration_sec REAL,
                error_message TEXT,
                output_path TEXT,
                
                FOREIGN KEY (session_id) REFERENCES sweep_sessions(id)
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id 
            ON sweep_runs(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_collective 
            ON sweep_runs(collective)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_num_nodes 
            ON sweep_runs(num_nodes)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_num_channels 
            ON sweep_runs(num_channels)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_algo 
            ON sweep_runs(algo)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_proto 
            ON sweep_runs(proto)
        """)
        
        # Per-message-size metrics table (granular data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sweep_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                
                -- Message details
                size_bytes INTEGER NOT NULL,
                count INTEGER NOT NULL,
                data_type TEXT NOT NULL,
                redop TEXT,
                
                -- Out-of-place metrics
                time_oop_us REAL,
                algbw_oop REAL,
                busbw_oop REAL,
                errors_oop INTEGER,
                
                -- In-place metrics
                time_ip_us REAL,
                algbw_ip REAL,
                busbw_ip REAL,
                errors_ip INTEGER,
                
                -- Algorithm/protocol/channels ACTUALLY SELECTED for this size,
                -- read from rccl-tests' -A 1 columns. RCCL substitutes silently, so
                -- these can differ from what was requested.
                algo TEXT,
                proto TEXT,
                nchannels INTEGER,

                -- GPU-107: what we asked for, so requested-vs-selected is auditable.
                requested_algo TEXT,
                requested_proto TEXT,
                requested_nchannels INTEGER,
                -- 1 = RCCL did NOT honour the request. Such rows must never be used
                -- to pick a winner: their label describes a config that never ran.
                substituted INTEGER DEFAULT 0,
                -- algo before stripping the RCCL addon marker (e.g. 'RING*' = WarpSpeed)
                measured_algo_raw TEXT,
                -- provenance of nchannels. 'A_flag_ceiling' = a planned ceiling, NOT what
                -- ran; real counts need NCCL_DEBUG=INFO channel{Lo..Hi}.
                nchannels_source TEXT,

                FOREIGN KEY (run_id) REFERENCES sweep_runs(id)
            )
        """)

        # Additive migration for databases created before the GPU-107 columns existed.
        cursor.execute("PRAGMA table_info(sweep_metrics)")
        _existing = {r[1] for r in cursor.fetchall()}
        for _col, _decl in (
            ('requested_algo', 'TEXT'),
            ('requested_proto', 'TEXT'),
            ('requested_nchannels', 'INTEGER'),
            ('substituted', 'INTEGER DEFAULT 0'),
            ('measured_algo_raw', 'TEXT'),
            ('nchannels_source', 'TEXT'),
        ):
            if _col not in _existing:
                cursor.execute(f"ALTER TABLE sweep_metrics ADD COLUMN {_col} {_decl}")
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_run_id 
            ON sweep_metrics(run_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_size 
            ON sweep_metrics(size_bytes)
        """)
        
        self.conn.commit()
    
    def create_session(self, config: Dict[str, Any], total_tests: int) -> int:
        """Create a new sweep session.
        
        Args:
            config: Configuration dictionary
            total_tests: Total number of tests planned
            
        Returns:
            Session ID
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO sweep_sessions (
                start_time, config_json, total_tests, status
            ) VALUES (?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            json.dumps(config),
            total_tests,
            'running'
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def update_session(self, session_id: int, **kwargs):
        """Update session information.
        
        Args:
            session_id: Session ID
            **kwargs: Fields to update
        """
        cursor = self.conn.cursor()
        
        updates = []
        values = []
        for key, value in kwargs.items():
            updates.append(f"{key} = ?")
            values.append(value)
        
        values.append(session_id)
        
        query = f"""
            UPDATE sweep_sessions 
            SET {', '.join(updates)}
            WHERE id = ?
        """
        
        cursor.execute(query, values)
        self.conn.commit()
    
    def complete_session(self, session_id: int):
        """Mark a session as completed.
        
        Args:
            session_id: Session ID
        """
        cursor = self.conn.cursor()
        
        # Get counts
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) as failed
            FROM sweep_runs WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        
        self.update_session(
            session_id,
            end_time=datetime.now().isoformat(),
            completed_tests=row['success'],
            failed_tests=row['failed'],
            status='completed'
        )
    
    def insert_run(self, run_data: Dict[str, Any]) -> int:
        """Insert a new test run.
        
        Args:
            run_data: Dictionary containing run information
            
        Returns:
            ID of inserted row
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO sweep_runs (
                session_id, timestamp, collective, num_nodes, num_gpus, num_channels,
                algo, proto,
                command, results_json, avg_busbw, max_busbw,
                rccl_version, hip_version, rocm_version,
                status, duration_sec, error_message, output_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_data.get('session_id'),
            run_data.get('timestamp', datetime.now().isoformat()),
            run_data.get('collective'),
            run_data.get('num_nodes'),
            run_data.get('num_gpus'),
            run_data.get('num_channels'),
            run_data.get('algo'),
            run_data.get('proto'),
            run_data.get('command'),
            run_data.get('results_json'),
            run_data.get('avg_busbw'),
            run_data.get('max_busbw'),
            run_data.get('rccl_version'),
            run_data.get('hip_version'),
            run_data.get('rocm_version'),
            run_data.get('status'),
            run_data.get('duration_sec'),
            run_data.get('error_message'),
            run_data.get('output_path')
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_metrics(self, run_id: int, metrics: List[Dict[str, Any]]) -> int:
        """Insert per-message-size metrics for a run.
        
        Args:
            run_id: ID of the parent run
            metrics: List of metric dictionaries from parser
            
        Returns:
            Number of metrics inserted
        """
        cursor = self.conn.cursor()
        
        for m in metrics:
            cursor.execute("""
                INSERT INTO sweep_metrics (
                    run_id, size_bytes, count, data_type, redop,
                    time_oop_us, algbw_oop, busbw_oop, errors_oop,
                    time_ip_us, algbw_ip, busbw_ip, errors_ip,
                    algo, proto, nchannels,
                    requested_algo, requested_proto, requested_nchannels,
                    substituted, measured_algo_raw, nchannels_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                m.get('size_bytes'),
                m.get('count'),
                m.get('type'),
                m.get('redop'),
                m.get('time_oop_us'),
                m.get('algbw_oop'),
                m.get('busbw_oop'),
                m.get('errors_oop'),
                m.get('time_ip_us'),
                m.get('algbw_ip'),
                m.get('busbw_ip'),
                m.get('errors_ip'),
                m.get('algo'),
                m.get('proto'),
                m.get('nchannels'),
                m.get('requested_algo'),
                m.get('requested_proto'),
                m.get('requested_nchannels'),
                m.get('substituted', 0),
                m.get('measured_algo_raw'),
                m.get('nchannels_source'),
            ))
        
        self.conn.commit()
        return len(metrics)
    
    def get_runs(self, session_id: Optional[int] = None, 
                 collective: Optional[str] = None,
                 num_nodes: Optional[int] = None,
                 num_channels: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get test runs with optional filtering.
        
        Args:
            session_id: Filter by session
            collective: Filter by collective type
            num_nodes: Filter by node count
            num_channels: Filter by channel count
            
        Returns:
            List of run dictionaries
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM sweep_runs WHERE 1=1"
        params = []
        
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        if collective is not None:
            query += " AND collective = ?"
            params.append(collective)
        if num_nodes is not None:
            query += " AND num_nodes = ?"
            params.append(num_nodes)
        if num_channels is not None:
            query += " AND num_channels = ?"
            params.append(num_channels)
        
        query += " ORDER BY timestamp"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        runs = []
        for row in rows:
            run = dict(row)
            if run.get('results_json'):
                run['metrics'] = json.loads(run['results_json'])
            runs.append(run)
        
        return runs
    
    def get_summary_stats(self, session_id: int) -> Dict[str, Any]:
        """Get summary statistics for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary with summary stats
        """
        cursor = self.conn.cursor()
        
        # Overall stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_runs,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) as failed,
                AVG(avg_busbw) as overall_avg_busbw,
                MAX(max_busbw) as overall_max_busbw,
                SUM(duration_sec) as total_duration
            FROM sweep_runs WHERE session_id = ?
        """, (session_id,))
        
        overall = dict(cursor.fetchone())
        
        # Per-collective stats
        cursor.execute("""
            SELECT 
                collective,
                COUNT(*) as runs,
                AVG(avg_busbw) as avg_busbw,
                MAX(max_busbw) as max_busbw
            FROM sweep_runs 
            WHERE session_id = ? AND status = 'success'
            GROUP BY collective
        """, (session_id,))
        
        per_collective = [dict(row) for row in cursor.fetchall()]
        
        # Per-node-count stats
        cursor.execute("""
            SELECT 
                num_nodes,
                num_gpus,
                COUNT(*) as runs,
                AVG(avg_busbw) as avg_busbw,
                MAX(max_busbw) as max_busbw
            FROM sweep_runs 
            WHERE session_id = ? AND status = 'success'
            GROUP BY num_nodes
            ORDER BY num_nodes
        """, (session_id,))
        
        per_node_count = [dict(row) for row in cursor.fetchall()]
        
        return {
            'overall': overall,
            'per_collective': per_collective,
            'per_node_count': per_node_count
        }
    
    def export_to_csv(self, output_path: str, session_id: Optional[int] = None):
        """Export results to CSV file.
        
        Args:
            output_path: Path to output CSV file
            session_id: Optional session ID to filter
        """
        import pandas as pd
        
        runs = self.get_runs(session_id=session_id)
        
        if not runs:
            print("No runs to export")
            return
        
        # Flatten for CSV
        rows = []
        for run in runs:
            row = {
                'timestamp': run['timestamp'],
                'collective': run['collective'],
                'num_nodes': run['num_nodes'],
                'num_gpus': run['num_gpus'],
                'num_channels': run['num_channels'],
                'algo': run.get('algo'),
                'proto': run.get('proto'),
                'avg_busbw': run['avg_busbw'],
                'max_busbw': run['max_busbw'],
                'status': run['status'],
                'duration_sec': run['duration_sec'],
                'rccl_version': run['rccl_version'],
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(rows)} runs to {output_path}")
    
    def get_metrics(self, run_id: Optional[int] = None, 
                    session_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get per-message-size metrics with optional filtering.
        
        Args:
            run_id: Filter by specific run
            session_id: Filter by session (gets metrics for all runs in session)
            
        Returns:
            List of metric dictionaries with run info
        """
        cursor = self.conn.cursor()
        
        query = """
            SELECT 
                m.*,
                r.collective,
                r.num_nodes,
                r.num_gpus,
                r.timestamp
            FROM sweep_metrics m
            JOIN sweep_runs r ON m.run_id = r.id
            WHERE 1=1
        """
        params = []
        
        if run_id is not None:
            query += " AND m.run_id = ?"
            params.append(run_id)
        if session_id is not None:
            query += " AND r.session_id = ?"
            params.append(session_id)
        
        query += " ORDER BY r.id, m.size_bytes"
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def export_metrics_to_csv(self, output_path: str, session_id: Optional[int] = None):
        """Export detailed per-message-size metrics to CSV file.
        
        Args:
            output_path: Path to output CSV file
            session_id: Optional session ID to filter
        """
        import pandas as pd
        
        metrics = self.get_metrics(session_id=session_id)
        
        if not metrics:
            print("No metrics to export")
            return
        
        # Flatten for CSV
        rows = []
        for m in metrics:
            row = {
                'collective': m['collective'],
                'num_nodes': m['num_nodes'],
                'num_gpus': m['num_gpus'],
                'size_bytes': m['size_bytes'],
                'count': m['count'],
                'data_type': m['data_type'],
                'redop': m['redop'],
                'time_oop_us': m['time_oop_us'],
                'algbw_oop': m['algbw_oop'],
                'busbw_oop': m['busbw_oop'],
                'errors_oop': m['errors_oop'],
                'time_ip_us': m['time_ip_us'],
                'algbw_ip': m['algbw_ip'],
                'busbw_ip': m['busbw_ip'],
                'errors_ip': m['errors_ip'],
                'algo': m['algo'],
                'proto': m['proto'],
                'nchannels': m['nchannels'],
            }
            # GPU-107: carry the requested-vs-measured audit trail into the CSV.
            # optimize_metrics.py reads `substituted` from here to exclude rows whose
            # request RCCL did not honour; without these keys the whole verification
            # chain silently degrades to the old requested-as-measured behaviour.
            for _k in ('requested_algo', 'requested_proto', 'requested_nchannels',
                       'substituted', 'measured_algo_raw', 'nchannels_source'):
                row[_k] = m[_k] if _k in m.keys() else None
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(rows)} metrics to {output_path}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == '__main__':
    # Test the database
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        
        with SweepDatabase(db_path) as db:
            # Create session
            session_id = db.create_session(
                config={'test': 'config'},
                total_tests=10
            )
            print(f"Created session: {session_id}")
            
            # Insert some runs
            for i in range(3):
                run_id = db.insert_run({
                    'session_id': session_id,
                    'collective': 'all_reduce_perf',
                    'num_nodes': 2,
                    'num_gpus': 16,
                    'num_channels': 4 * (i + 1),
                    'command': 'mpirun ...',
                    'avg_busbw': 250.0 + i * 10,
                    'max_busbw': 380.0 + i * 5,
                    'status': 'success',
                    'duration_sec': 120.0
                })
                print(f"Inserted run: {run_id}")
            
            # Get runs
            runs = db.get_runs(session_id=session_id)
            print(f"Retrieved {len(runs)} runs")
            
            # Get summary
            summary = db.get_summary_stats(session_id)
            print(f"Summary: {summary['overall']}")
            
            # Complete session
            db.complete_session(session_id)
            print("Session completed")

