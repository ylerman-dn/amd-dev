#!/usr/bin/env python3
"""
Combination Validator Module

Validates RCCL collective/algo/proto/node combinations against known
unsupported configurations defined in unsupported_combos.yaml.
"""

import os
import yaml
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional


class ComboValidator:
    """
    Validates RCCL test combinations against known unsupported configurations.
    
    Usage:
        validator = ComboValidator()  # Uses default config
        is_valid, reason = validator.is_supported(
            collective='all_reduce',
            algo='TREE',
            proto='SIMPLE',
            num_nodes=1
        )
        if not is_valid:
            print(f"Skipping: {reason}")
    """
    
    # Default config path (relative to this module)
    DEFAULT_CONFIG = Path(__file__).parent.parent / 'unsupported_combos.yaml'
    
    def __init__(self, config_path: Optional[Path] = None, rccl_build: Optional[str] = None):
        """
        Initialize the validator.

        Args:
            config_path: Path to unsupported_combos.yaml. If None, uses default.
            rccl_build: identifier of the RCCL build in use, e.g.
                "gfx950 / RCCL 2.28.3-develop:2e42aa8". Rules carrying a
                `measured_on:` field are EMPIRICAL -- observed on one machine and
                one build rather than derived from tuning.cc -- and are skipped
                with a warning when this does not match. Pass None to keep every
                rule (previous behaviour), or set RCCL_BUILD_ID in the
                environment.

        Why this exists: a wrong empirical rule is invisible. A skipped
        combination leaves no trace in the results, so a rule that was true on
        gfx950/2.28.3 and false elsewhere silently removes a candidate that might
        have won, and nothing in the output would ever say so.
        """
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG
        self.rccl_build = rccl_build if rccl_build is not None else os.environ.get("RCCL_BUILD_ID")
        self.skipped_rules: List[Dict[str, Any]] = []
        self.rules: List[Dict[str, Any]] = []
        self._load_rules()
    
    def _load_rules(self) -> None:
        """Load rules from YAML config file."""
        if not self.config_path.exists():
            # No config file - all combinations allowed
            return
        
        try:
            with open(self.config_path) as f:
                data = yaml.safe_load(f)
            
            if data and 'unsupported' in data:
                all_rules = data['unsupported'] or []
                if self.rccl_build is None:
                    self.rules = all_rules
                else:
                    for r in all_rules:
                        measured = r.get('measured_on')
                        if measured and measured != self.rccl_build:
                            self.skipped_rules.append(r)
                        else:
                            self.rules.append(r)
                    if self.skipped_rules:
                        print(f"Note: {len(self.skipped_rules)} empirical combo rule(s) skipped -- "
                              f"measured on a different build than {self.rccl_build!r}. "
                              f"Those combinations will now be swept. "
                              f"Re-measure and update measured_on: to re-enable them.")
        except Exception as e:
            print(f"Warning: Could not load combo rules from {self.config_path}: {e}")
            self.rules = []
    
    def _normalize_collective(self, collective: str) -> str:
        """Normalize collective name (remove _perf suffix, lowercase)."""
        return collective.lower().replace('_perf', '').replace('-', '_')
    
    def _normalize_algo(self, algo: str) -> str:
        """Normalize algorithm name (uppercase)."""
        return algo.upper()
    
    def _normalize_proto(self, proto: str) -> str:
        """Normalize protocol name (uppercase)."""
        return proto.upper()
    
    def _matches_rule(
        self,
        rule: Dict[str, Any],
        collective: str,
        algo: str,
        proto: str,
        num_nodes: int
    ) -> bool:
        """
        Check if a combination matches a rule.
        
        Args:
            rule: Rule dictionary from config
            collective: Collective type (e.g., 'all_reduce')
            algo: Algorithm (e.g., 'TREE')
            proto: Protocol (e.g., 'SIMPLE')
            num_nodes: Number of nodes
            
        Returns:
            True if the combination matches this rule (i.e., is unsupported)
        """
        # Check collective
        if 'collective' in rule:
            rule_coll = self._normalize_collective(rule['collective'])
            if rule_coll != 'any' and rule_coll != collective:
                return False
        
        # Check algorithm
        if 'algo' in rule:
            rule_algo = self._normalize_algo(rule['algo'])
            if rule_algo != 'ANY' and rule_algo != algo:
                return False
        
        # Check protocol
        if 'proto' in rule:
            rule_proto = self._normalize_proto(rule['proto'])
            if rule_proto != 'ANY' and rule_proto != proto:
                return False
        
        # Check max_nodes constraint
        if 'max_nodes' in rule:
            if num_nodes > rule['max_nodes']:
                return False
        
        # Check min_nodes constraint
        if 'min_nodes' in rule:
            if num_nodes < rule['min_nodes']:
                return False
        
        # All conditions matched - this rule applies
        return True
    
    def is_supported(
        self,
        collective: str,
        algo: str,
        proto: str,
        num_nodes: int
    ) -> Tuple[bool, str]:
        """
        Check if a combination is supported.
        
        Args:
            collective: Collective type (e.g., 'all_reduce', 'all_gather_perf')
            algo: Algorithm (e.g., 'TREE', 'RING')
            proto: Protocol (e.g., 'SIMPLE', 'LL', 'LL128')
            num_nodes: Number of nodes
            
        Returns:
            Tuple of (is_supported, reason).
            If supported: (True, "")
            If not supported: (False, "reason string")
        """
        # Normalize inputs
        collective = self._normalize_collective(collective)
        algo = self._normalize_algo(algo)
        proto = self._normalize_proto(proto)
        
        # Check against all rules
        for rule in self.rules:
            if self._matches_rule(rule, collective, algo, proto, num_nodes):
                reason = rule.get('reason', 'Unsupported combination')
                return False, reason
        
        return True, ""
    
    def get_unsupported_reason(
        self,
        collective: str,
        algo: str,
        proto: str,
        num_nodes: int
    ) -> Optional[str]:
        """
        Get the reason why a combination is unsupported.
        
        Returns:
            Reason string if unsupported, None if supported.
        """
        is_valid, reason = self.is_supported(collective, algo, proto, num_nodes)
        return reason if not is_valid else None
    
    def filter_supported_combinations(
        self,
        combinations: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], str]]]:
        """
        Filter a list of combinations into supported and unsupported.
        
        Args:
            combinations: List of dicts with keys: collective, algo, proto, num_nodes
            
        Returns:
            Tuple of (supported_list, unsupported_list_with_reasons)
        """
        supported = []
        unsupported = []
        
        for combo in combinations:
            is_valid, reason = self.is_supported(
                combo.get('collective', ''),
                combo.get('algo', ''),
                combo.get('proto', ''),
                combo.get('num_nodes', 1)
            )
            
            if is_valid:
                supported.append(combo)
            else:
                unsupported.append((combo, reason))
        
        return supported, unsupported


def main():
    """Command-line interface for testing the validator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test RCCL combination validator'
    )
    parser.add_argument(
        '--collective', '-c',
        default='all_reduce',
        help='Collective type'
    )
    parser.add_argument(
        '--algo', '-a',
        default='RING',
        help='Algorithm'
    )
    parser.add_argument(
        '--proto', '-p',
        default='SIMPLE',
        help='Protocol'
    )
    parser.add_argument(
        '--nodes', '-n',
        type=int,
        default=1,
        help='Number of nodes'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to unsupported_combos.yaml'
    )
    
    args = parser.parse_args()
    
    validator = ComboValidator(args.config)
    
    is_valid, reason = validator.is_supported(
        args.collective,
        args.algo,
        args.proto,
        args.nodes
    )
    
    print(f"Collective: {args.collective}")
    print(f"Algorithm:  {args.algo}")
    print(f"Protocol:   {args.proto}")
    print(f"Nodes:      {args.nodes}")
    print(f"Supported:  {'Yes' if is_valid else 'No'}")
    if not is_valid:
        print(f"Reason:     {reason}")


if __name__ == '__main__':
    main()

